import torch
import torch.nn.functional as F
import copy

def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    """
    top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。
    Args:
        logits: 预测结果，即预测成为词典中每个词的分数
        top_k: 只保留概率最高的top_k个标记
        top_p: 只保留概率累积达到top_p的标记
        filter_value: 过滤标记值

    Returns:

    """
    # logits的维度必须为2，即size:[batch_size, vocab_size]
    assert logits.dim() == 2
    # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
    top_k = min(top_k, logits[0].size(-1))
    # 如果top_k不为0，则将在logits中保留top_k个标记
    if top_k > 0:
        # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的top_k标记
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value
    # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
    if top_p > 0.0:
        # 对logits进行递减排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # 对排序后的结果使用softmax归一化，再获取累积概率序列
        # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 删除累积概率高于top_p的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，使第一个标记也保持在top_p之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的累积概率达到top_p的标记
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def predict_one_sample(model, tokenizer, device, args, content):
    """
    对单个样本进行预测
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备信息
        args: 配置项信息
        content: 新闻正文

    Returns:

    """
    # 对新闻正文进行预处理，并判断如果超长则进行截断
    content.strip("\n")
    content_tokens = tokenizer.tokenize(content)
    if len(content_tokens) > args.max_len - 3 - args.generate_max_len:
        content_tokens = content_tokens[:(args.max_len - 3 - args.generate_max_len)//2] + content_tokens[len(content_tokens)-(args.max_len - 3 - args.generate_max_len)//2:]
    # 获取content_id、title_id、unk_id、sep_id值
    content_id = tokenizer.convert_tokens_to_ids("[Content]")
    title_id = tokenizer.convert_tokens_to_ids("[Title]")
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    # 将tokens索引化，变成模型所需格式
    content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    # 将input_ids和token_type_ids进行扩充，扩充到需要预测标题的个数，即batch_size
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
    token_type_ids = [[content_id] * len(content_tokens) for _ in range(args.batch_size)]
    # 将input_ids和token_type_ids变成tensor
    input_tensors = torch.tensor(input_ids).long().to(device)
    token_type_tensors = torch.tensor(token_type_ids).long().to(device)
    next_token_type = torch.tensor([[title_id] for _ in range(args.batch_size)]).long().to(device)
    # 用于存放每一步解码的结果
    generated = []
    # 用于存放，完成解码序列的序号
    finish_set = set()
    with torch.no_grad():
        # 遍历生成标题最大长度
        for _ in range(args.generate_max_len):
            outputs = model(input_ids=input_tensors, token_type_ids=token_type_tensors)
            # 获取预测结果序列的最后一个标记，next_token_logits size：[batch_size, vocab_size]
            next_token_logits = outputs[0][:, -1, :]
            # 对batch_size进行遍历，将词表中出现在序列中的词的概率进行惩罚
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            # 对batch_size进行遍历，将词表中的UNK的值设为无穷小
            for next_token_logit in next_token_logits:
                next_token_logit[unk_id] = -float("Inf")
            # 使用top_k_top_p_filtering函数，按照top_k和top_p的值，对预测结果进行筛选
            filter_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            # 对filter_logits的每一行做一次取值，输出结果是每一次取值时filter_logits对应行的下标，即词表位置（词的id）
            # filter_logits中的越大的值，越容易被选中
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            # 判断如果哪个序列的预测标记为sep_id时，则加入到finish_set
            for index, token_id in enumerate(next_tokens[:, 0]):
                if token_id == sep_id:
                    finish_set.add(index)
            # 判断，如果finish_set包含全部的序列序号，则停止预测；否则继续预测
            finish_flag = True
            for index in range(args.batch_size):
                if index not in finish_set:
                    finish_flag = False
                    break
            if finish_flag:
                break
            # 将预测标记添加到generated中
            generated.append([token.item() for token in next_tokens[:, 0]])
            # 将预测结果拼接到input_tensors和token_type_tensors上，继续下一次预测
            input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
            token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)
        # 用于存储预测结果
        candidate_responses = []
        # 对batch_size进行遍历，并将token_id变成对应汉字
        for index in range(args.batch_size):
            responses = []
            for token_index in range(len(generated)):
                # 判断，当出现sep_id时，停止在该序列中添加token
                if generated[token_index][index] != sep_id:
                    responses.append(generated[token_index][index])
                else:
                    break
            # 将token_id序列变成汉字序列，去除"##"，并将[Space]替换成空格
            candidate_responses.append(
                "".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[space]", " "))
    return candidate_responses


