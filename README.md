## 1. 실행
### 1.0. 데이터 다운로드
제공된 파일을 res 디렉토리에 다운로드 받습니다.

```bash
├── model
   ├── arena_data
      └── results.json 
   ├── train.py
   ├── inference.py
├── res
   ├── song_meta.json
   ├── train.json
   ├── val.json
   └── test.json
```

### 1.1. 모델 학습 (w2v)
train.py를 실행하면 모델 파일이 동일 폴더에 생성됩니다.

```bash
$> python train.py 
```

#### 1.2. 평가 데이터 생성
inference.py을 실행하면 model/arena_data/results.json 결과 파일이 생성됩니다.

```bash
$> python inference.py
```
