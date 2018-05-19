import os
import io
from logging import getLogger
import numpy as np
from scipy.stats import spearmanr
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

SEMEVAL17_EVAL_PATH = 'data/crosslingual/wordsim'
DIC_EVAL_PATH = 'data/crosslingual/dictionaries'

logger = getLogger()

def get_crosslingual_wordsim_scores(lang1, word2id1, embeddings1, lang2, word2id2, embeddings2, lower=True):

    f1 = os.path.join(SEMEVAL17_EVAL_PATH, '%s-%s-SEMEVAL17.txt' % (lang1, lang2))
    f2 = os.path.join(SEMEVAL17_EVAL_PATH, '%s-%s-SEMEVAL17.txt' % (lang2, lang1))
    if not (os.path.exists(f1) or os.path.exists(f2)):
        return None

    if os.path.exists(f1):
        coeff, found, not_found = get_spearman_rho(
            word2id1, embeddings1, f1,
            lower, word2id2, embeddings2
        )
    elif os.path.exists(f2):
        coeff, found, not_found = get_spearman_rho(
            word2id2, embeddings2, f2,
            lower, word2id1, embeddings1
        )

    return coeff


def get_spearman_rho(word2id1, embeddings1, path, lower, word2id2=None, embeddings2=None):
    
    assert not ((word2id2 is None) ^ (embeddings2 is None))
    word2id2 = word2id1 if word2id2 is None else word2id2
    embeddings2 = embeddings1 if embeddings2 is None else embeddings2
    assert type(lower) is bool
    word_pairs = get_word_pairs(path)
    not_found = 0
    pred = []
    gold = []
    for word1, word2, similarity in word_pairs:
        id1 = get_word_id(word1, word2id1, lower)
        id2 = get_word_id(word2, word2id2, lower)
        if id1 is None or id2 is None:
            not_found += 1
            continue
        u = embeddings1[id1]
        v = embeddings2[id2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)
    return spearmanr(gold, pred).correlation, len(gold), not_found


def get_word_pairs(path, lower=True):
    
    assert os.path.isfile(path) and type(lower) is bool
    word_pairs = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            line = line.lower() if lower else line
            line = line.split()
            # ignore phrases, only consider words
            if len(line) != 3:
                assert len(line) > 3
                assert 'SEMEVAL17' in os.path.basename(path) or 'EN-IT_MWS353' in path
                continue
            word_pairs.append((line[0], line[1], float(line[2])))
    return word_pairs


def get_word_id(word, word2id, lower):
    assert type(lower) is bool and lower==True
    word_id = word2id.get(word)
    return word_id


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method):
    
    path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    dico = load_dictionary(path, word2id1, word2id2).cuda()

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)
    
    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    
    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # cross-domain similarity local scaling
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        k = int(method[len('csls_knn_'):])
        average_dist1 = get_knn_avg_dist_faiss(emb2, emb1, k)
        average_dist2 = get_knn_avg_dist_faiss(emb1, emb2, k)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None] + average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(10, 1, largest=True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        matching = {}
        matching_unique = {}
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
            matching_unique[src_id.item()] = min(matching_unique.get(src_id.item(), 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        precision_at_k_unique = 100 * np.mean(list(matching_unique.values()))
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        logger.info("%i unique source words - %s - Precision at k = %i: %f" %
                    (len(matching_unique), method, k, precision_at_k_unique))
        results.append(('precision_at_%i' % k, precision_at_k))
        results.append(('precision_at_%i_unique' % k, precision_at_k_unique))

    return results


def get_unsupervised_evaluation(word2id1, emb1, word2id2, emb2, knn):
    
    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    
    average_dist1 = get_knn_avg_dist_faiss(emb2, emb1, knn)
    average_dist2 = get_knn_avg_dist_faiss(emb1, emb2, knn)
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
    
    scores = emb1[:5000].mm(emb2.transpose(0, 1))
    scores.mul_(2)
    scores.sub_(average_dist1[:5000][:, None] + average_dist2[None, :])
    
    top_matches = scores.topk(1, 1, largest=True)[0]
    
    return torch.mean(top_matches).item()

    
def get_knn_avg_dist_faiss(emb, query, knn):
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()
    

def load_dictionary(path, word2id1, word2id2):
    
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower(), "uppercase found"
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("found %i pairs of words in the dictionary. "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico