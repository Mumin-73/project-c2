import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# eSpeak NG 라이브러리 경로 설정
_ESPEAK_LIBRARY = 'C:/Program Files/eSpeak NG/libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)

# 현재 스크립트의 경로를 기준으로 설정
output_path = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 경로 설정
metadata_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/project_c2/TTS/TTS/tts/datasets/my_datasets/metadata.csv"
dataset_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/project_c2/TTS/TTS/tts/datasets/my_datasets"

# 데이터셋 설정
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train=metadata_path,
    path=dataset_path
)

# 학습 설정 초기화
config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="ja",  # 일본어로 설정
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    phonemizer="espeak",
)

# 오디오 프로세서 초기화
ap = AudioProcessor.init_from_config(config)

# 토크나이저 초기화
tokenizer, config = TTSTokenizer.init_from_config(config)

# 데이터 샘플 로드
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

trainer.fit()