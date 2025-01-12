def store_clips_to_file(clips, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for clip in clips:
            f.write(clip + '\n\n')
