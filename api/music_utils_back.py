# import os
# # 音频分离器模型缓存目录
# os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = "audio-separator-models"

# from audio_separator.separator import Separator

## 分离伴奏和歌声
# def do_s(audio):

#     # Initialize the Separator class (with optional configuration properties, below)
#     separator = Separator()

#     # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
#     separator.load_model()

#     # Perform the separation on specific audio files without reloading the model
#     output_files = separator.separate(audio)

#     return output_files[1],output_files[0]


## 合并伴奏和歌声
# def do_m(audio, back):


#     print(audio)

#     write("output.wav", audio[0], audio[1])

#     # 加载背景音乐和人声文件
#     background_music = AudioSegment.from_file(back)
#     vocal = AudioSegment.from_file("output.wav")

#     # 合并音频文件
#     combined_audio = background_music.overlay(vocal)

#     # 导出合并后的音频文件
#     combined_audio.export("combined_audio.wav", format="wav")

#     return "combined_audio.wav","combined_audio.wav"