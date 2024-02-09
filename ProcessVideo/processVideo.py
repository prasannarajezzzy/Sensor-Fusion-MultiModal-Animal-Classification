import os
import moviepy.editor as mp
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import imageio

def generate_spectrogram(audio_path, output_folder, clip_number):
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    spectrogram_path = os.path.join(output_folder, f'spectrogram_clip_{clip_number}.png')
    plt.tight_layout()
    plt.savefig(spectrogram_path)
    plt.close()
    return spectrogram_path

def save_middle_frame(video_clip, output_folder, clip_number):
    mid_frame = video_clip.duration / 2
    mid_frame_image_path = os.path.join(output_folder, f'mid_frame_clip_{clip_number}.png')
    mid_frame_image = video_clip.get_frame(mid_frame)
    imageio.imwrite(mid_frame_image_path, mid_frame_image)
    return mid_frame_image_path

def process_video(video_path, output_folder):
    video_clip = mp.VideoFileClip(video_path)

    # Create folders for saving files
    os.makedirs(output_folder, exist_ok=True)

    # Split the video into 10-second intervals
    intervals = np.arange(0, video_clip.duration, 10)
    for i, start_time in enumerate(intervals):
        end_time = min(start_time + 10, video_clip.duration)

        # Extract audio from the video clip within the current interval
        audio_clip = video_clip.audio.subclip(start_time, end_time)
        audio_path = os.path.join(output_folder, f'audio_clip_{i}.wav')
        audio_clip.write_audiofile(audio_path, codec='pcm_s16le', fps=44100)

        # Generate spectrogram for the audio clip
        generate_spectrogram(audio_path, output_folder, i)

        # Save the middle frame of the 10-second clip
        mid_frame_image_path = save_middle_frame(video_clip.subclip(start_time, end_time), output_folder, i)

        print(f'Mid-frame and spectrogram generated for clip {i + 1}: {mid_frame_image_path}')

if __name__ == "__main__":
    video_path = '/content/Terrifying Elephant Roar.mp4'  # Replace with the path to your video file
    output_folder = 'data'  # Replace with the desired output folder
    process_video(video_path, output_folder)
