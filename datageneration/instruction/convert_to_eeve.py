"""
checkpoint 파일을 EEVE 템플릿 형식으로 변환
messages 형태 -> text 형태
"""

import json
from pathlib import Path
from datetime import datetime


def convert_to_eeve_format(input_file: str, output_file: str = None):
    """messages 형태를 EEVE 템플릿 형태로 변환"""
    
    print(f"\n변환 시작: {input_file}")
    
    # 입력 파일 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 로드 완료: {len(data):,}개 샘플")
    
    # 변환
    converted_data = []
    for i, item in enumerate(data, 1):
        if 'messages' in item:
            # messages에서 user와 assistant 추출
            user_content = None
            assistant_content = None
            
            for msg in item['messages']:
                if msg['role'] == 'user':
                    user_content = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_content = msg['content']
            
            if user_content and assistant_content:
                # EEVE 템플릿 형식으로 변환
                eeve_text = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                    f"Human: {user_content}\n"
                    f"Assistant: {assistant_content}<|im_end|>"
                )
                
                converted_data.append({
                    'text': eeve_text,
                    'metadata': item.get('metadata', {})
                })
            else:
                print(f"  ⚠️  샘플 {i}: user 또는 assistant 내용 없음")
        elif 'text' in item:
            # 이미 EEVE 형식이면 그대로 유지
            converted_data.append(item)
        else:
            print(f"  ⚠️  샘플 {i}: 알 수 없는 형식")
        
        if i % 5000 == 0:
            print(f"  진행: {i:,} / {len(data):,}")
    
    print(f"✓ 변환 완료: {len(converted_data):,}개 샘플")
    
    # 출력 파일명 생성
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.parent / f"{input_path.stem}_eeve_{timestamp}.json"
    
    # JSON 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON 저장: {output_file}")
    
    # JSONL 저장
    output_jsonl = Path(output_file).with_suffix('.jsonl')
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ JSONL 저장: {output_jsonl}")
    
    # 샘플 출력
    print("\n" + "="*80)
    print("변환 샘플 예시:")
    print("="*80)
    if converted_data:
        sample = converted_data[0]
        text = sample['text']
        # Human과 Assistant 부분만 출력
        if 'Human:' in text and 'Assistant:' in text:
            parts = text.split('Human:', 1)[1].split('Assistant:', 1)
            print(f"\nHuman: {parts[0].strip()[:200]}...")
            print(f"\nAssistant: {parts[1].strip()[:300]}...")
        print(f"\nMetadata: {sample['metadata']}")
    
    return output_file, output_jsonl


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert checkpoint to EEVE format')
    parser.add_argument('input_file', type=str, help='입력 checkpoint 파일')
    parser.add_argument('--output', type=str, default=None, help='출력 파일명 (선택)')
    
    args = parser.parse_args()
    
    convert_to_eeve_format(args.input_file, args.output)


if __name__ == "__main__":
    main()
