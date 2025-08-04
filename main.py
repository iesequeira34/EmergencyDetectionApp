import os
from fastapi import FastAPI, UploadFile
import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition
import random
from faster_whisper import WhisperModel

import re
import shutil

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch.set_default_device("cuda")
SPEAKER_PATH = "./speaker_voices/"
EMBEDDING_MODEL = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",  run_opts={"device":"cuda"})
STT_MODELX = WhisperModel(model_size_or_path="base.en", device="cuda", compute_type="float16")
UPLOADS = "./uploads/"

ONBOARDING_SENTENCES = {
    1: "the quick brown fox jumps over the lazy dog",
    2: "today is a beautiful day outside",
    3: "artificial intelligence is the future"
}

app = FastAPI()

@app.get("/health")
def check_health():
    return {"healthStatus": f"{True}"}

@app.get("/check_enroll/{user_id}")
def isEnrollmentDone(user_id: int):
    user_id_path = os.path.join(SPEAKER_PATH, str(user_id))
    if os.path.exists(user_id_path) and len(os.listdir(user_id_path)) == 3: 
        return {"enrollStatus": True}
    else:
        return {"enrollStatus": False}

@app.post("/enroll/{user_id}/{sentence_id}")
async def enroll_user(uploaded_file: UploadFile, user_id: int, sentence_id: int):
    user_id_path = os.path.join(SPEAKER_PATH, str(user_id))
    # max_len = len(os.listdir(user_id_path)) if os.path.exists(user_id_path) else 0
    # new_file = os.path.join(user_id_path, f"{max_len+1}.wav")

    # if len(os.listdir(user_id_path)) == 3:
    #     shutil.rmtree(user_id_path)

    new_file = os.path.join(user_id_path, f"{sentence_id}.wav")
    
    if not isEnrollmentDone(user_id=user_id)['enrollStatus']:
        if not os.path.exists(user_id_path):
            os.makedirs(user_id_path)
        try:
            content = await uploaded_file.read()
            with open(new_file, 'wb') as wavfile:
                wavfile.write(content)
            segments, _ = STT_MODELX.transcribe(new_file, beam_size=7, vad_filter=True)
            output_text = " ".join(segment.text for segment in segments)
            output_text = re.sub("[^A-Za-z\s]", "", output_text)
            output_text = re.sub("\s\s+", " ", output_text)
            print("User said: ", output_text)
            if ONBOARDING_SENTENCES[sentence_id] in output_text.lower():
                return {"uploadStatus": "Success"}
            else:
                return {"uploadStatus": "Audio does not match sentence. Please try again"}
        except Exception as e:
            if os.path.exists(new_file):
                os.remove(new_file)
            return {"Error": f"Could not enroll UserID={user_id} with the file: {str(e)}"}
    else:
        return {"uploadStatus": f"Enrollment already done for UserID={user_id}. File was ignored."}
    


@app.post("/emergency_detection/{user_id}")
async def emergency_detection(uploaded_file: UploadFile, user_id: int):
    new_file = os.path.join(UPLOADS, f"{user_id}_emd_{random.randint(1,10000)}.wav")
    if not os.path.exists(UPLOADS):
        os.makedirs(UPLOADS)
    if not isEnrollmentDone(user_id=user_id)['enrollStatus']:
        return {"Error": f"You need to enroll UserID={user_id} with 3 samples of your voice. Only then wake word detection can be done."}
    else:
        try:
            content = await uploaded_file.read()
            with open(new_file, 'wb') as wavfile:
                wavfile.write(content)

            segments, _ = STT_MODELX.transcribe(new_file, beam_size=7, vad_filter=True)
            output_text = " ".join(segment.text for segment in segments)
            output_text = re.sub("[^A-Za-z\s]", "", output_text)
            output_text = re.sub("\s\s+", " ", output_text)
            user_id_path = os.path.join(SPEAKER_PATH, str(user_id))
            score_list = [EMBEDDING_MODEL.verify_files(path_x=os.path.join(user_id_path, file), path_y=new_file)[0] for file in os.listdir(user_id_path)]
            final_score = np.mean([score.cpu() for score in score_list])
            wakeword_detected = "help" in output_text.strip().lower()
            # emergency_detected = wakeword_detected and (final_score >= 0.2)
            emergency_detected = wakeword_detected

            print(f"User said: {output_text.strip()}, Emergency: {emergency_detected}, User match: {final_score}")
            return {"IsEmergency": f"{emergency_detected}"}
        except Exception as e:
            if os.path.exists(new_file):
                os.remove(new_file)
            print(str(e))
            return {"Error": f"Could not detect emergency UserID={user_id} as: {str(e)}"}
        finally:
            if os.path.exists(new_file):
                os.remove(new_file)

