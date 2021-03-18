import subprocess

cmd = ['ffmpeg', '-y', '-ss', '00:00:10.000', '-i', '1.m3u8', '-vframes', '1', '-vf', 'scale=720:480',
                   'out.jpg']
            devnull = open(os.devnull, 'wb')
            subprocess.Popen(cmd, cwd="/", stdout=devnull, stderr=devnull)
            print(cmd)