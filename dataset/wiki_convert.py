import pickle
import os
import requests
import time
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

def load_mappings(dataset_folder: str, domain: str) -> Dict[str, Dict]:
    """載入指定領域的映射檔案"""
    mapping_path = os.path.join(dataset_folder, domain, 'og_mappings.pkl')
    
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    
    return mappings

def get_wikidata_label(qid: str, lang: str = 'en') -> str:
    """
    從 Wikidata API 獲取實體標籤
    
    Args:
        qid: Wikidata ID (如 Q1765268)
        lang: 語言代碼
    
    Returns:
        str: 實體標籤，如果無法獲取則返回原始 QID
    """
    try:
        url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'props': 'labels',
            'languages': lang
        }
        response = requests.get(url, params=params, timeout=10).json()
        label = response['entities'][qid]['labels'].get(lang, {}).get('value', None)
        return label if label else qid
    except Exception as e:
        print(f"無法獲取 {qid} 的標籤: {e}")
        return qid

def batch_get_wikidata_labels(qids: List[str], lang: str = 'en', batch_size: int = 50) -> Dict[str, str]:
    """
    批量獲取 Wikidata 標籤
    
    Args:
        qids: Wikidata ID 列表
        lang: 語言代碼
        batch_size: 批量大小
    
    Returns:
        Dict[str, str]: QID 到標籤的映射
    """
    labels = {}
    
    for i in tqdm(range(0, len(qids), batch_size), desc="Getting Wikidata Labels"):
        batch_qids = qids[i:i + batch_size]
        try:
            url = 'https://www.wikidata.org/w/api.php'
            params = {
                'action': 'wbgetentities',
                'ids': '|'.join(batch_qids),
                'format': 'json',
                'props': 'labels',
                'languages': lang
            }
            response = requests.get(url, params=params, timeout=30).json()
            
            for qid in batch_qids:
                if qid in response['entities']:
                    label = response['entities'][qid]['labels'].get(lang, {}).get('value', None)
                    labels[qid] = label if label else None
                else:
                    labels[qid] = None
                    
            # 避免請求過於頻繁
            time.sleep(0.1)
            
        except Exception as e:
            print(f"批量獲取標籤失敗 (batch {i//batch_size + 1}): {e}")
            # 如果批量失敗，嘗試單個獲取
            for qid in batch_qids:
                single_label = get_wikidata_label(qid, lang)
                labels[qid] = single_label if single_label != qid else None
                time.sleep(0.1)
    
    return labels

def decode_graph_file(file_path: str, mappings: Dict, train_wikidata_labels: Dict[str, str], test_wikidata_labels: Dict[str, str], relation_labels: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """解碼圖檔案，過濾掉無法找到標籤的三元組"""
    if 'train' in file_path or 'val' in file_path:
        id2e = {v: k for k, v in mappings['e2id_train'].items()}
        wikidata_labels = train_wikidata_labels
    else:
        id2e = {v: k for k, v in mappings['e2id_test'].items()}
        wikidata_labels = test_wikidata_labels
    id2r = {v: k for k, v in mappings['r2id'].items()}
    
    decoded_triples = []    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                head_id, relation_id, tail_id = map(int, line.strip().split())
                
                # 檢查頭實體
                head_name = id2e.get(head_id, f"Unknown_Entity_{head_id}")
                if head_name.startswith("Unknown_Entity_"):
                    continue
                head_label = wikidata_labels.get(head_name)
                if not head_label:
                    continue
                
                # 檢查關係
                relation_name = id2r.get(relation_id, f"Unknown_Relation_{relation_id}")
                if relation_name.startswith("Unknown_Relation_"):
                    continue
                relation_label = relation_labels.get(relation_name)
                if not relation_label:
                    continue
                
                # 檢查尾實體
                tail_name = id2e.get(tail_id, f"Unknown_Entity_{tail_id}")
                if tail_name.startswith("Unknown_Entity_"):
                    continue
                tail_label = wikidata_labels.get(tail_name)
                if not tail_label:
                    continue
                
                decoded_triples.append((head_label, relation_label, tail_label))
    
    return decoded_triples

def process_domain(dataset_folder: str, domain: str, output_folder: str):
    """
    處理指定領域的所有相關檔案
    
    Args:
        dataset_folder: WikiTopics_QE 資料夾路徑
        domain: 領域名稱
        output_folder: 輸出資料夾路徑
    """
    print(f"\n=== 處理 {domain} 領域 ===")
    
    # 建立輸出資料夾
    domain_output = os.path.join(output_folder, domain)
    os.makedirs(domain_output, exist_ok=True)
    mappings_output = os.path.join(domain_output, 'mappings')
    os.makedirs(mappings_output, exist_ok=True)
    
    # 載入映射
    mappings = load_mappings(dataset_folder, domain)
    
    # 收集所有需要獲取標籤的 Wikidata ID
    print("收集 Wikidata IDs...")
    train_entity_names = set(mappings['e2id_train'].keys())
    test_entity_names = set(mappings['e2id_test'].keys())
    relation_names = set(mappings['r2id'].keys())
    
    # 過濾出 Wikidata ID (以 Q 開頭的)
    train_wikidata_ids = [name for name in train_entity_names]
    test_wikidata_ids = [name for name in test_entity_names]
    
    # 過濾出關係 ID (以 P 開頭的，去除 _inv 後綴)
    relation_ids = set()
    for name in relation_names:
        if name.endswith('_inv'):
            name = name[:-4]  # 移除 '_inv'
        relation_ids.add(name)
    relation_ids = list(relation_ids)
    
    # 批量獲取 Wikidata 標籤
    print(f"找到 {len(train_wikidata_ids)} 個 Train Wikidata IDs，開始獲取標籤...")
    train_wikidata_labels = batch_get_wikidata_labels(train_wikidata_ids)
    print(f"找到 {len(test_wikidata_ids)} 個 Test Wikidata IDs，開始獲取標籤...")
    test_wikidata_labels = batch_get_wikidata_labels(test_wikidata_ids)
    print(f"找到 {len(relation_ids)} 個關係 IDs，開始獲取標籤...") 
    relation_labels = batch_get_wikidata_labels(relation_ids)
    
    # 儲存實體映射
    print("儲存實體映射...")
    train_entity_mapping_path = os.path.join(mappings_output, 'train_entity_mappings.txt')
    with open(train_entity_mapping_path, 'w', encoding='utf-8') as f:
        f.write("ID\tOriginal_Name\tWikidata_Label\n")
        for entity_name, entity_id in mappings['e2id_train'].items():
            wikidata_label = train_wikidata_labels.get(entity_name)
            if wikidata_label:  # 只保存有標籤的
                f.write(f"{entity_id}\t{entity_name}\t{wikidata_label}\n")

    test_entity_mapping_path = os.path.join(mappings_output, 'test_entity_mappings.txt')
    with open(test_entity_mapping_path, 'w', encoding='utf-8') as f:
        f.write("ID\tOriginal_Name\tWikidata_Label\n")
        for entity_name, entity_id in mappings['e2id_test'].items():
            wikidata_label = test_wikidata_labels.get(entity_name)
            if wikidata_label:  # 只保存有標籤的
                f.write(f"{entity_id}\t{entity_name}\t{wikidata_label}\n")
    
    # 儲存關係映射
    print("儲存關係映射...")
    relation_mapping_path = os.path.join(mappings_output, 'relation_mappings.txt')
    with open(relation_mapping_path, 'w', encoding='utf-8') as f:
        f.write("ID\tOriginal_Name\tWikidata_Label\n")
        for relation_name, relation_id in mappings['r2id'].items():
            wikidata_label = relation_labels.get(relation_name)

            if relation_name.endswith('_inv'):
                base_name = relation_name[:-4]
                base_label = relation_labels.get(base_name)
                if base_label:
                    wikidata_label = f"{base_label} (inverse)"
                else:
                    continue 

            if wikidata_label:
                f.write(f"{relation_id}\t{relation_name}\t{wikidata_label}\n")
    
    # 處理 graph 檔案
    print("處理 graph 檔案...")
    graph_file_paths = ["train_graph.txt", "val_inference.txt", "test_inference.txt"]
    for graph_file in graph_file_paths:
        graph_path = os.path.join(dataset_folder, domain, graph_file)
        if os.path.exists(graph_path):
            print(f"處理 {graph_file}...")
            decoded_triples = decode_graph_file(graph_path, mappings, train_wikidata_labels, test_wikidata_labels, relation_labels)
            
            output_path = os.path.join(domain_output, graph_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Head\tRelation\tTail\n")
                for head, relation, tail in decoded_triples:
                    f.write(f"{head}\t{relation}\t{tail}\n")
    
    # 為 query 和 answer 準備映射檔案
    relation_mappings, test_entity_mappings = {}, {}
    with open(relation_mapping_path, "r") as f:
        relation_mappings_list = f.readlines()[1:]  # 跳過標題行
        relation_mappings_list = [line.strip().split("\t") for line in relation_mappings_list]
        relation_mappings = {line[0]: line[2] for line in relation_mappings_list}
    with open(train_entity_mapping_path, "r") as f:
        train_entity_mappings_list = f.readlines()[1:]  # 跳過標題行
        train_entity_mappings_list = [line.strip().split("\t") for line in train_entity_mappings_list]
        train_entity_mappings = {line[0]: line[2] for line in train_entity_mappings_list}
    with open(test_entity_mapping_path, "r") as f:
        test_entity_mappings_list = f.readlines()[1:]   # 跳過標題行
        test_entity_mappings_list = [line.strip().split("\t") for line in test_entity_mappings_list]
        test_entity_mappings = {line[0]: line[2] for line in test_entity_mappings_list}
    
    # 處理 query 檔案
    print("處理 query 檔案...")
    query_file_paths = ["train_queries.pkl", "valid_queries.pkl", "test_queries.pkl"]
    query_type = ('e', ('r', 'r', 'r'))
    for query_file in query_file_paths:
        if "test" in query_file:
            entity_mappings = test_entity_mappings
        else:
            entity_mappings = train_entity_mappings

        query_path = os.path.join(dataset_folder, domain, query_file)
        with open(query_path, 'rb') as f:
            data = pickle.load(f)[query_type]

            entities = []
            for key in data:
                (e, (r1, r2, r3)) = key                
                e = entity_mappings.get(f'{e}', None)
                r1 = relation_mappings.get(f'{r1}', None)
                r2 = relation_mappings.get(f'{r2}', None)
                r3 = relation_mappings.get(f'{r3}', None)
                if e is None or r1 is None or r2 is None or r3 is None:
                    continue
                query = f"('{e.replace("'", "\\\'")}', ('{r1.replace("'", "\\\'")}', '{r2.replace("'", "\\\'")}', '{r3.replace("'", "\\\'")}'))"
                entities.append(query)

            with open(os.path.join(domain_output, query_file.replace('.pkl', '.json')), "w") as f:
                json.dump({"('e', ('r', 'r', 'r'))": entities}, f, indent=2, ensure_ascii=False)
    
    # 處理 answer 檔案
    print("處理 answer 檔案...")
    answer_file_paths = ["train_answers_hard.pkl", "valid_answers_easy.pkl", "valid_answers_hard.pkl", "test_answers_easy.pkl", "test_answers_hard.pkl"]
    query_type = ('e', ('r', 'r', 'r'))
    for answer_file in answer_file_paths:
        if "test" in answer_file:
            entity_mappings = test_entity_mappings
        else:
            entity_mappings = train_entity_mappings

        answer_path = os.path.join(dataset_folder, domain, answer_file)
        with open(answer_path, 'rb') as f:
            data = pickle.load(f)[query_type]
            
            answers = {}
            for key, value in data.items():
                (e, (r1, r2, r3)) = key            
                e = entity_mappings.get(f'{e}', None)
                r1 = relation_mappings.get(f'{r1}', None)
                r2 = relation_mappings.get(f'{r2}', None)
                r3 = relation_mappings.get(f'{r3}', None)
                if e is None or r1 is None or r2 is None or r3 is None:
                    continue
                answer_key = f"('{e.replace("'", "\\\'")}', ('{r1.replace("'", "\\\'")}', '{r2.replace("'", "\\\'")}', '{r3.replace("'", "\\\'")}'))"

                values = []
                for v in value:
                    v = entity_mappings.get(f'{v}', None)
                    if v is None:
                        continue
                    values.append(v)

                answers[answer_key] = values

            with open(os.path.join(domain_output, answer_file.replace('.pkl', '.json')), "w") as f:
                json.dump({f"{query_type}": answers}, f, indent=2, ensure_ascii=False)

# 主要執行函數
def main():
    # 設定路徑
    dataset_folder = "WikiTopics_QE"  # 請修改為你的實際路徑
    output_folder = "WikiTopicsQE_decoded"  
    domains = ['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax']
    
    for domain in tqdm(domains, desc="Processing Domains"):
        try:
            process_domain(dataset_folder, domain, output_folder)
            print(f"\n{'='*50}\n")
        except Exception as e:
            print(f"處理 {domain} 時發生錯誤: {e}")
            continue

if __name__ == "__main__":
    main()