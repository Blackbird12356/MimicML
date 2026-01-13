"""
gesture_gui.py

Tkinter GUI для записи/обучения/распознавания жестов (MediaPipe -> KNN).
Папки и файлы создаются автоматически:
 - gestures_dataset/  (субпапки: gesture_name/ с .npy файлами, каждый файл = один временной сэмпл)
 - gifs/              (если gif выбран, можно хранить копию там, но по умолчанию сохраняем путь)
 - gesture_labels.json
 - gesture_model.pkl
"""

import os, time, json, threading, glob, pickle
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from tkinter import *
from tkinter import simpledialog, messagebox, filedialog
from PIL import Image, ImageTk, ImageSequence
from sklearn.neighbors import KNeighborsClassifier

# Настройки
DATA_DIR = "gestures_dataset"
GIFS_DIR = "gifs"
LABELS_JSON = "gesture_labels.json"
MODEL_PATH = "gesture_model.pkl"

# сколько секунд записывать пример жеста
RECORD_SECONDS = 2.0
# частота кадров (сколько кадров сохраняем за секунду)
SAMPLE_FPS = 15
# сколько примеров собирать по умолчанию при добавлении жеста (можно повторять)
SAMPLES_PER_GESTURE = 8

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GIFS_DIR, exist_ok=True)

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Работа с данными
def load_labels():
    if os.path.exists(LABELS_JSON):
        return json.load(open(LABELS_JSON, "r", encoding="utf-8"))
    return {}  # id -> {"name":..., "gif": path}

def save_labels(labels):
    json.dump(labels, open(LABELS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def gesture_folder(name):
    safe = name.replace(" ", "_")
    folder = os.path.join(DATA_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    return folder

def list_gestures():
    lbls = load_labels()
    # return ordered list of (id, name)
    return [(k, v["name"]) for k, v in lbls.items()]

#  Извлечение признаков (landmarks -> вектор)
# Мы берём: Pose landmarks (33) + Hands (21*2) = 33 + 42 = 75 точек -> координаты x,y,z -> 225 dims
# Но не всегда обе руки видны — в этом случае заполняем нулями.
# Для инвариантности по положению/размеру, нормализуем координаты относительно туловища (плечо/позы).

def landmarks_to_vector(results):
    """
    results: MediaPipe holistic result
    возвращаем 1D numpy array фиксированной длины
    """
    # Pose: 33 landmarks, each has x,y,z, visibility
    pose = results.pose_landmarks.landmark if results.pose_landmarks else None
    # Hands: left and right
    left = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    right = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None


    vec = []

    # helper to append N landmarks (x,y,z) or zeros
    def append_landmarks(lm_list, expected_count):
        if lm_list is None:
            vec.extend([0.0] * (expected_count * 3))
        else:
            # записываем первые expected_count
            for i in range(expected_count):
                if i < len(lm_list):
                    lm = lm_list[i]
                    vec.extend([lm.x, lm.y, lm.z])
                else:
                    vec.extend([0.0, 0.0, 0.0])

    # pose 33
    append_landmarks(pose, 33)
    # left hand 21, right hand 21
    append_landmarks(left, 21)
    append_landmarks(right, 21)

    arr = np.array(vec, dtype=np.float32)

    # выбираем опорную точку: pose[0] (нос) или плечи (11 - левое плечо)
    if pose:
        # используем среднее плечей если доступны
        try:
            left_sh = pose[11]
            right_sh = pose[12]
            cx = (left_sh.x + right_sh.x) / 2.0
            cy = (left_sh.y + right_sh.y) / 2.0
        except:
            # fallback nose
            nose = pose[0]
            cx, cy = nose.x, nose.y
    else:
        cx, cy = 0.0, 0.0

    # сдвигаем x,y в массиве
    arr2 = arr.reshape(-1, 3).copy()
    arr2[:, 0] -= cx
    arr2[:, 1] -= cy
    # масштабирование
    maxv = np.max(np.abs(arr2[:, :2])) if arr2.size else 1.0
    if maxv < 1e-6:
        maxv = 1.0
    arr2[:, :2] /= maxv
    return arr2.flatten()

# Сбор (запись) примеров жеста
def record_gesture_samples(gesture_name, samples=SAMPLES_PER_GESTURE, duration=RECORD_SECONDS):
    """
    Записывает samples примеров, каждый длится duration секунд, возвращает количество успешно сохранённых файлов.
    Сохраняет npy файлы в gestures_dataset/<gesture_name>/
    """
    folder = gesture_folder(gesture_name)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Камера", "Камера не найдена.")
        return 0

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    saved = 0

    try:
        for s in range(samples):
            messagebox.showinfo("Запись", f"Подготовься к показу жеста '{gesture_name}'. Запись начнётся по нажатию ОК.\nПример {s+1}/{samples}")
            frames_needed = int(duration * SAMPLE_FPS)
            seq = []
            count = 0
            last_time = 0.0
            start = time.time()
            while count < frames_needed:
                ok, frame = cap.read()
                if not ok:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                vec = landmarks_to_vector(results)
                # сохраняем каждую N-й итерацию, чтобы получить ~SAMPLE_FPS
                # просто собираем последовательность
                seq.append(vec)
                count += 1
                # визуальная подсказка
                cv2.putText(frame, f"Recording {gesture_name}  ({s+1}/{samples})  frame {count}/{frames_needed}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imshow("Recording gesture (press q to cancel)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt("User cancelled")
            # после записи последовательности: сохраняем как npy
            fn = os.path.join(folder, f"{gesture_name}_{int(time.time()*1000)}.npy")
            np.save(fn, np.array(seq))  # shape = (frames_needed, feat_dim)
            saved += 1
    except KeyboardInterrupt:
        print("Recording cancelled")
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()
    return saved

# Подготовка тренировочного набора
def load_dataset_for_training():
    X = []
    y = []
    labels = load_labels()
    for gid, info in labels.items():
        name = info["name"]
        folder = gesture_folder(name)
        for fn in glob.glob(os.path.join(folder, "*.npy")):
            arr = np.load(fn)
            mean_vec = np.mean(arr, axis=0)
            std_vec = np.std(arr, axis=0)
            feat = np.concatenate([mean_vec, std_vec])
            X.append(feat)
            y.append(int(gid))
    if len(X) == 0:
        return None, None
    return np.vstack(X), np.array(y, dtype=np.int32)

#  Обучение модели
def train_gesture_model():
    X, y = load_dataset_for_training()
    if X is None:
        messagebox.showwarning("Данные", "Нет примеров для обучения. Сначала добавьте жесты.")
        return False
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    messagebox.showinfo("Готово", f"Модель обучена на {len(y)} примерах ({len(set(y))} жестов).")
    return True

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Распознавание в реальном времени
class VideoThread:
    def __init__(self, video_label, gif_label):
        self.video_label = video_label
        self.gif_label = gif_label
        self._stop = threading.Event()
        self.thread = None
        self.current_gesture = None
        self.frames_for_gif = []
        self.gif_frames = []
        self.model = None
        self.labels = load_labels()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        # clear gif
        self.gif_frames = []
        self.frames_for_gif = []
        self.current_gesture = None

    def _run(self):
        # load model
        self.model = load_model()
        if self.model is None:
            messagebox.showwarning("Модель", "Модель не обучена. Сначала обучите.")
            return
        self.labels = load_labels()
        cap = cv2.VideoCapture(0)
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        seq_buffer = []
        frames_window = int(SAMPLE_FPS * 1.2)  # скользящие окна
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(img_rgb)
                vec = landmarks_to_vector(results)
                seq_buffer.append(vec)
                if len(seq_buffer) > frames_window:
                    seq_buffer.pop(0)
                # формируем признаковый вектор (mean+std) из текущего буфера
                if len(seq_buffer) >= max(3, frames_window//2):
                    arr = np.vstack(seq_buffer)
                    feat = np.concatenate([np.mean(arr, axis=0), np.std(arr, axis=0)]).reshape(1, -1)
                    pred = self.model.predict(feat)[0]
                    # вероятность? у KNN нет predict_proba по умолчанию для metric='minkowski' возможно есть
                    # Для стабильности требуем повторяющиеся предсказания:
                    # если предсказание отличается — требуем устойчивость: текущий жест должен держаться N итераций
                    # реализуем простую логику: если pred == current -> ok; else заменяем immed.
                    if self.current_gesture != pred:
                        # смена
                        self.current_gesture = pred
                        self._load_gif_for_gesture(pred)
                # рисуем landmarks поверх фрейма (для наглядности)
                annotated = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # отображаем видеокадр в video_label
                img_disp = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_disp)
                imgtk = ImageTk.PhotoImage(img_pil.resize((480, 360)))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # если есть gif frames - проигрываем
                if self.gif_frames:
                    # прокручиваем
                    f = self.gif_frames.pop(0)
                    self.gif_frames.append(f)
                    self.gif_label_display(f)
                else:
                    # очистка gif label
                    self.gif_label_display(None)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            holistic.close()
            cap.release()
            cv2.destroyAllWindows()

    def _load_gif_for_gesture(self, gid):
        # загрузим gif указанный в labels[gid]["gif"]
        self.gif_frames = []
        self.frames_for_gif = []
        info = self.labels.get(str(gid))
        if not info:
            return
        path = info.get("gif")
        if not path or not os.path.exists(path):
            # no gif assigned
            return
        # load frames via PIL
        try:
            im = Image.open(path)
            frames = []
            for frame in ImageSequence.Iterator(im):
                frame = frame.convert("RGBA")
                # scale down
                frame = frame.resize((250, 250))
                frames.append(ImageTk.PhotoImage(frame))
            self.gif_frames = frames.copy()
        except Exception as e:
            print("GIF load error:", e)
            self.gif_frames = []

    def gif_label_display(self, imgtk):
        # imgtk is PhotoImage or None
        if imgtk is None:
            self.gif_label.config(image="")
        else:
            self.gif_label.config(image=imgtk)

#  Tkinter GUI
root = Tk()
root.title("Gesture Trainer & Recognizer")
root.geometry("900x520")

left = Frame(root, width=260)
left.pack(side=LEFT, fill=Y, padx=8, pady=8)

Label(left, text="Gestures", font=("Arial", 14)).pack(pady=6)
gesture_listbox = Listbox(left, width=30, height=18)
gesture_listbox.pack(padx=4)

def refresh_gesture_list():
    gesture_listbox.delete(0, END)
    lbls = load_labels()
    for k, v in lbls.items():
        gesture_listbox.insert(END, f"{k}: {v['name']}")

def add_gesture():
    name = simpledialog.askstring("New Gesture", "Enter gesture name (no spaces recommended):")
    if not name:
        return
    # create label entry
    labels = load_labels()
    new_id = max([int(x) for x in labels.keys()] + [0]) + 1 if labels else 1
    labels[str(new_id)] = {"name": name, "gif": ""}
    save_labels(labels)
    refresh_gesture_list()
    # record samples
    saved = record_gesture_samples(name, samples=SAMPLES_PER_GESTURE, duration=RECORD_SECONDS)
    if saved:
        messagebox.showinfo("Saved", f"Saved {saved} samples for '{name}'. Now choose a GIF (optional).")
    else:
        messagebox.showwarning("No data", "No samples saved.")
    # prompt for gif
    if messagebox.askyesno("GIF", "Choose GIF file for this gesture?"):
        change_gif_for_selected(name_key=str(new_id))

def change_gif_for_selected(name_key=None):
    labels = load_labels()
    if name_key is None:
        sel = gesture_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Select a gesture first.")
            return
        text = gesture_listbox.get(sel[0])
        gid = text.split(":")[0]
    else:
        gid = name_key
    file = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
    if not file:
        return
    labels = load_labels()
    if gid in labels:
        labels[gid]["gif"] = file
        save_labels(labels)
        messagebox.showinfo("GIF", "GIF associated with gesture.")
    refresh_gesture_list()

def delete_gesture():
    sel = gesture_listbox.curselection()
    if not sel:
        messagebox.showwarning("Select", "Select a gesture to delete.")
        return
    text = gesture_listbox.get(sel[0])
    gid = text.split(":")[0]
    labels = load_labels()
    name = labels.get(gid, {}).get("name", gid)
    if not messagebox.askyesno("Delete", f"Delete gesture '{name}' and all its samples?"):
        return
    # delete folder samples
    folder = gesture_folder(name)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        try:
            os.rmdir(folder)
        except:
            pass
    labels.pop(gid, None)
    save_labels(labels)
    # remove model to require re-train
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    refresh_gesture_list()

Button(left, text="Add Gesture (record)", width=24, command=add_gesture).pack(pady=6)
Button(left, text="Change GIF", width=24, command=change_gif_for_selected).pack(pady=6)
Button(left, text="Delete Gesture", width=24, command=delete_gesture).pack(pady=6)
Button(left, text="Train Model", width=24, command=train_gesture_model).pack(pady=12)

# right: video + gif
right = Frame(root)
right.pack(side=RIGHT, expand=True, fill=BOTH, padx=8, pady=8)

video_label = Label(right)
video_label.pack(padx=6, pady=6)

gif_label = Label(right)
gif_label.pack(padx=6, pady=6)

vt = VideoThread(video_label, gif_label)

def start_recognize():
    # ensure model exists
    if not os.path.exists(MODEL_PATH):
        if not messagebox.askyesno("No model", "Model not trained. Train now?"):
            return
        train_gesture_model()
    vt.start()

def stop_recognize():
    vt.stop()

Button(right, text="Start Recognize", width=16, command=start_recognize).pack(pady=4)
Button(right, text="Stop", width=16, command=stop_recognize).pack(pady=4)

refresh_gesture_list()

def on_closing():
    vt.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
