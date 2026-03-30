#!/usr/bin/env python3
import wave
import json
import sys
import os

def inspect(path):
    if not os.path.exists(path):
        print(json.dumps({"error": "not_found", "path": path}))
        return 2
    try:
        with wave.open(path, 'rb') as wf:
            info = {
                "path": path,
                "nchannels": wf.getnchannels(),
                "sampwidth": wf.getsampwidth(),
                "framerate": wf.getframerate(),
                "nframes": wf.getnframes(),
                "comptype": wf.getcomptype(),
                "compname": wf.getcompname(),
            }
            print(json.dumps(info))
            return 0
    except wave.Error as e:
        print(json.dumps({"error": "wave_error", "message": str(e)}))
        return 1
    except Exception as e:
        print(json.dumps({"error": "exception", "message": str(e)}))
        return 1

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'Eucalyptus_arcana.wav'
    sys.exit(inspect(path))
