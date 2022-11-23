from __future__ import absolute_import, division, print_function #保证模型精度


from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import torch
from tqdm import tqdm, trange
from random import random, shuffle, choice, sample
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import os
import logging
import pretrained_arg as args


gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)#环境的驱动
logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def warmup_linear(x, warmup=0.001):
    if x < warmup:
        return x/warmup
    return 1.0-x

def create_mask_prediction(tokens, mask_proportion, vocab_list, mask_prediction_max_sequence):
    candidate_index = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        candidate_index.append(i)
    num_mask = min(mask_prediction_max_sequence, max(1, int(round(len(tokens)*mask_proportion))))
    shuffle(candidate_index)
    mask_index = sorted(sample(candidate_index, num_mask))
    mask_tokens = []
    for i in mask_index:
        if random() < 0.8:#简便 避免使用计数器
            mask_token = "[MASK]"
        else:
            if random() < 0.5:
                mask_token = tokens[i]
            else:
                mask_token = choice(vocab_list)
        mask_tokens.append(tokens[i])
        tokens[i] = mask_token
    return tokens, mask_index, mask_tokens



def create_mask_examples(data_path, mask_prediction_max_sequence, max_length, mask_proportion, tokenizer, vocab_list):
    #为什么要在这里传vacab_list进来？随机替换就是从vocab_list中random选择的。
    #vocab_list 在预训练的时候使用它，是为了提供一个词库（原语料）。
    #mask_prediction_max_sequence   BERT没说 一般设置为20 30

    mask_examples = []
    max_num_token = max_length - 2
    f = open(data_path, "r", encoding="utf8")
    #k = 0
    for (i, line) in tqdm(enumerate(f), desc="mask creating"):
        line = line.strip()
        line = line.replace("\u2028", "")
        tokens = tokenizer.tokenize(line.strip())[:max_num_token]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_id = [0 for _ in range(len(tokens))]
        if len(tokens) <= 32:
            continue
        tokens, masked_lm_positions, masked_lm_labels = create_mask_prediction(tokens=tokens, mask_proportion=mask_proportion, vocab_list=vocab_list,
                               mask_prediction_max_sequence=mask_prediction_max_sequence)
        example = {"tokens": tokens,
                   "segment_ids": segment_id,
                   "masked_lm_positions": masked_lm_positions,
                   "masked_lm_labels": masked_lm_labels}
        mask_examples.append(example)
    f.close()
    return mask_examples

def convert_examples_to_features(examples, max_sequence_length, tokenizer):
    features = []
    for i, example in tqdm(enumerate(examples), desc="make feature"):
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        assert len(tokens) == len(segment_ids) <= max_sequence_length
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        input_array = np.zeros(max_sequence_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids #padding
        mask_array = np.zeros(max_sequence_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1
        segment_array = np.zeros(max_sequence_length, dtype=bool)
        segment_array[:len(segment_ids)] = segment_ids
        lm_label_array = np.full(max_sequence_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids
        feature = InputFeatures(input_ids=input_array, input_mask=mask_array, segment_ids=segment_array, label_id=lm_label_array)
        features.append(feature)
    return features




def main():
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        num_gpu = torch.cuda.device_count()
    else:

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        num_gpu = 1
        torch.distributed.init_process_group(backend="nccl")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size//args.gradient_accumulation_steps
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file)
    train_examples = None
    num_train_opt_steps = None
    vocab_list = []
    with open(args.vocab_file, "r", encoding="utf8") as f:
        for lines in f:
            vocab_list.append(lines.strip("\n"))

    if args.do_train:
        train_examples = []
        for _ in range(args.dupe_factor):
            instances = create_mask_examples(data_path=args.pretrained_path, mask_prediction_max_sequence=args.mask_prediction_max_sequence,
                                             max_length=args.max_length, mask_proportion=args.mask_proportion,
                                             tokenizer=tokenizer, vocab_list=vocab_list)
            train_examples += instances
        num_opt_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.train_epoch
        if args.local_rank != -1:
            num_opt_steps = num_opt_steps // torch.distributed.get_world_size()

    model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    if args.init_model != "":
        model.from_pretrained(args.init_model)
    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
             from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif num_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.frozen:
        for p in model.bert.embeddings.word_embeddings.parameters():
            p.requires_grad = False
        parameters_opt = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    else:
        parameters_opt = list(model.named_parameters())#冻结了一部分，就只优化部分，逐一检查标志位，filter

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]#为什么LN层不参加训练，归一化并不需要参加训练
    optimizer_grouped_parameters = [
    {'params': [p for n, p in parameters_opt if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in parameters_opt if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, bias_correction=False, max_grad_norm=1.0)

        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                             t_total=num_opt_steps)
    global_steps = 0
    best_loss = 1000000
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, args.max_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        nb_tr_steps = 0
        for e in trange(int(args.train_epoch), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples = 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if num_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        lr_this_step = args.lr * warmup_linear(global_steps / num_train_opt_steps, args.warmup_proportion)
                        for p in optimizer.param_groups:
                            p["lr"] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1
                if nb_tr_steps > 0 and nb_tr_steps % 100 == 0:
                    logger.info("==================== -epoch %d -train_step %d -train_loss %.4f\n" % (e, nb_tr_steps, tr_loss/nb_tr_steps))
                if nb_tr_steps > 0 and nb_tr_steps % args.save_checkpoint_steps == 0 and not args.do_eval:
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
            if nb_tr_steps > 0 and args.do_eval:
                eval_examples = create_mask_examples(data_path=args.pretrained_eval_path, mask_prediction_max_sequence=args.mask_prediction_max_sequence,
                                                     max_length=args.max_length, mask_proportion=args.mask_proportion,
                                                     tokenizer=tokenizer, vocab_list=vocab_list)
                eval_features = convert_examples_to_features(examples=eval_examples, max_sequence_length=args.max_length,
                                                             tokenizer=tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="start eval"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    with torch.no_grad():
                        loss = model(input_ids, segment_ids, input_mask, label_ids)
                    eval_loss += loss.item()
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps
                if eval_loss < best_loss:
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_loss = eval_loss
                logger.info("==================== -epoch %d -train_loss %.4f -eval_loss %.4f\n" % (
                e, tr_loss / nb_tr_steps, eval_loss))


if __name__ == "__main__":
    main()



