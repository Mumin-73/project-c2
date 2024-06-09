import os
import multiprocessing
import logging
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import Dataset
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
    output_path = os.path.dirname(os.path.abspath(__file__))
    metadata_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/metadata.csv"
    dataset_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets"

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=metadata_path,
        path=dataset_path
    )

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
        phoneme_language="ja",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )

    # Add logging configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_script")

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Update the get_phonemes method of the Dataset class
    def get_phonemes(self, idx, text):
        try:
            out_dict = self.tokenizer.text_to_ids(text, language=self.phoneme_language)
            assert len(out_dict["token_ids"]) > 0, f"Empty token IDs for text: {text}"
            return out_dict
        except AssertionError as e:
            logger.error(f"Error processing text at index {idx}: {str(e)}")
            raise e

    Dataset.get_phonemes = get_phonemes

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )

    # Remove the existing experiment folder if it exists
    experiment_folder = os.path.join(output_path, "run-May-27-2024_08+22PM-dbf1a08a")
    if os.path.exists(experiment_folder):
        logger.info(f"Removing existing experiment folder: {experiment_folder}")
        try:
            import shutil
            shutil.rmtree(experiment_folder)
        except Exception as e:
            logger.error(f"Error removing experiment folder: {str(e)}")

    trainer.fit()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()