"""
プロセス開始時にspawnを指定
（macbookでは動きません。画像のパスが違います）
"""
# ライブラリインポート
import multiprocessing as mp
from multiprocessing import Process, Manager, Value
import pickle
import math
import time
import sys
import traceback
import random

from PIL import Image, ImageDraw, ImageFont
import cv2
import face_recognition
import dlib
import numpy as np

import easing
from objects import face_frame_v2
from objects import similar_window, line_v3


# databaseの読み込み
print("start indexing")
datas = {}
with open('big_data.pickle', mode='rb') as f:
    datas = pickle.load(f)
# databese配列の作成
face_image_names = []
face_vectors = []
for k in datas:
    face_image_names.append(k)
    face_vectors.append(datas[k])
face_vectors = np.array(face_vectors).astype("float32")

# annoy
from annoy import AnnoyIndex

t = AnnoyIndex(face_vectors.shape[1])
for i, v in enumerate(face_vectors):
    t.add_item(i, v)

t.build(10)


# faissを用いたPQ
#nlist = 100

#m = 8
#k = 8  # 類似顔7こほしいのでk=8
#d = 128  # 顔特徴ベクトルの次元数
#quantizer = faiss.IndexFlatL2(d)  # this remains the same
#index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#index.train(face_vectors)
#index.add(face_vectors)
print("indexing is end")

def recommend_faces(similar_paths_manager, frame_manager, face_rect_manager, similar_distance_manager):
    """
    カメラ映像から取得した人物の類似顔を探し出す関数
    """
    while True:
        # frame manager が送られてきていなければcontinue
        if not frame_manager:
            continue

        # numpyへの変換
        try:
            frame = frame_manager[0]
        except OSError:
            print("OSerror occured")
        frame = np.array(frame)

        # 顔認識機のインスタンス
        detector = dlib.get_frontal_face_detector()

        # 顔データ(人数分)
        rects = detector(frame, 1)

        # 顔認識できなかったときcontinue
        # 共有メモリ内のデータも消す
        if not rects:
            face_rect_manager[:] = []
            similar_paths_manager[:] = []
            similar_distance_manager[:] = []
            print("cant recognize faces")
            continue

        dsts = []
        # face_rect_managerには複数人数になっても顔特徴ベクトルがそのまま入る
        if face_rect_manager[:] == []:
            for rect in rects:
                dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
                dsts.append(dst)
                rect_size = [rect.top(), rect.bottom(), rect.left(), rect.right()]
                face_rect_manager.append(rect_size)
        else:
            face_rect_manager[:] = []
            for x, rect in enumerate(rects):
                dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
                dsts.append(dst)
                rect_size = [rect.top(), rect.bottom(), rect.left(), rect.right()]
                face_rect_manager.append(rect_size)


        # 距離測定(人数分)
        # 顔情報のベクトル化　類似配列の生成
        similar_paths_manager[:] = []
        similar_distance_manager[:] = []
        D = []
        for i in range(len(dsts)):
            try:
                target_image_encoded = face_recognition.face_encodings(dsts[i])[0]
            except IndexError:
                continue

            target_vector = np.array(list(target_image_encoded)).astype("float32")
            target_vector.resize((1, 128))

            similar_paths = []
            similar_distances = []
            r, d = t.get_nns_by_vector(target_vector, 8, include_distances=True)
            for i, v in enumerate(r):
                similar_paths.append(face_image_names[v])
                similar_distances.append(d[i])
            print("finish about one face")
            # 画像パスの保存(何らかの問題あり)
            similar_paths_manager.append(similar_paths)
            similar_distance_manager.append(similar_distances)


if __name__ == '__main__':
    # マネージャの作成
    with Manager() as manager:
        # マネージャーの作成
        similar_paths_manager = manager.list()
        similar_distance_manager = manager.list()
        frame_manager = manager.list()
        face_rect_manager = manager.list()
        # プロセスの生成
        # 開始モードの指定
        ctx = mp.get_context("spawn")
        recommend_process = ctx.Process(target=recommend_faces, args=[similar_paths_manager, frame_manager, face_rect_manager, similar_distance_manager],
                                    name="recommend")
        # プロセスの開始
        recommend_process.start()
        start_time = time.time()
        cap = cv2.VideoCapture('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)RGB ! '
               'videoconvert ! appsink'.format(0, 128, 128), cv2.CAP_GSTREAMER)  # 引数はカメラのデバイス番号

        # cap.set(6,cv2.VideoWriter_fourcc(*'MJPG')) # 対応していない模様
        # パラメータを最大以上に上げてしまうとバグが発生する
        cap.set(5,30) # fps
        cap.set(4,1944) # height
        cap.set(3,2592) # width


        # 撮影の開始
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # 配列への変換/共有メモリへの代入
            if frame_manager[:] == []:
                print(cap)
                print(frame)
                frame_manager.append(list(frame))
            else:
                frame_manager[0] = list(frame)

            # まだ結果が出ていないなら
            if not similar_paths_manager:
                frame = frame[:, :, ::-1].copy()
                frame = Image.fromarray(frame)
                # frame = frame.resize((frame.width*2, frame.height*2))
                # cv2への変換
                frame=np.asarray(frame)
                frame = frame[:, :, ::-1]
                cv2.imshow('FaceMandara', frame)
                k = cv2.waitKey(1)
                continue

            # 類似顔を入れておく配列
            # similar_paths_managerをPILオブジェクトにした（v8_2との相違点）
            # imagesは一つのターゲットについての検索結果が入る変数。
            # all_imagesは動画内のターゲットすべてについての検索結果
            all_images = []
            try:
                for i in range(len(similar_paths_manager)):
                    images = []
                    for j in range(len(similar_paths_manager[i])):
                        images.append(Image.open("./big_database/{}".format(similar_paths_manager[i][j])))
                    all_images.append(images)
            except:
                print("something occured1")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

            # 顔認識部分の読み込み
            # face_rect_managerを名前付けしてやっただけ
            rects = []
            try:
                for i in range(len(face_rect_manager)):
                    rect = []
                    # print("i", i)
                    rect.append(face_rect_manager[i][0])# top
                    rect.append(face_rect_manager[i][1]) # bottom
                    rect.append(face_rect_manager[i][2]) # left
                    rect.append(face_rect_manager[i][3]) # right
                    rects.append(rect)
            except:
                print("something occured2")

            if similar_distance_manager:
                try:
                    distance = similar_distance_manager[0]
                    distance_no1 = distance[0][0]
                    distance_no2 = distance[0][1]
                    distance_no3 = distance[0][2]
                    distance_worst = distance[0][-1]
                except:
                    pass

            # インスタンス作成
            # similar_windows = 二重リスト
            if len(face_rect_manager) != len(all_images):
                continue

            similar_windows = []
            lines = []
            print("len all_images", len(all_images))
            for i in range(len(face_rect_manager)):
                similar_windows_one_rect = []
                lines_for_one_rect = []
                for j in range(len(all_images[i])):
                    print("j", j)
                    if len(face_rect_manager) != 1:
                        # 真下
                        if  j == 0:
                            amount_movement_x = 0
                            amount_movement_y = 180
                        # 真左
                        elif j == 1:
                            amount_movement_x = -210
                            amount_movement_y = 0
                        # 真右
                        elif j == 2:
                            amount_movement_x = 210
                            amount_movement_y = 0
                        elif j== 3:
                            amount_movement_x = 150
                            amount_movement_y = 140
                        elif j == 4:
                            amount_movement_x = -150
                            amount_movement_y = 140
                        elif j == 5:
                            amount_movement_x = 120
                            amount_movement_y = 20
                        else:
                            amount_movement_x = -120
                            amount_movement_y = 20
                        sw = similar_window.SimilarWindow(distance=distance_no1, \
                        place=[0, 0], image=all_images[i][j], \
                        similar_num=len(all_images[i])-j, \
                        movement_amount=(amount_movement_x, amount_movement_y), \
                        lonly = False)

                    else:
                        # 真下
                        if  j == 0:
                            amount_movement_x = 0
                            amount_movement_y = 360
                        # 真左
                        elif j == 1:
                            amount_movement_x = -420
                            amount_movement_y = 0
                        # 真右
                        elif j == 2:
                            amount_movement_x = 420
                            amount_movement_y = 0
                        elif j== 3:
                            amount_movement_x = 300
                            amount_movement_y = 280
                        elif j == 4:
                            amount_movement_x = -300
                            amount_movement_y = 280
                        elif j == 5:
                            amount_movement_x = 240
                            amount_movement_y = 40
                        else:
                            amount_movement_x = -240
                            amount_movement_y = 40
                        sw = similar_window.SimilarWindow(distance=distance_no1, \
                        place=[0, 0], image=all_images[i][j], \
                        similar_num=len(all_images[i])-j, \
                        movement_amount=(amount_movement_x, amount_movement_y), \
                        lonly = True)


                    similar_windows_one_rect.append(sw)
                    li = line_v3.Line(movement_amount_x=amount_movement_x, movement_amount_y=amount_movement_y)
                    lines_for_one_rect.append(li)
                similar_windows.append(similar_windows_one_rect)
                lines.append(lines_for_one_rect)

            face_frames = []
            # face_frames = []
            for i in range(len(rects)):
                ff = face_frame_v2.FaceFrame(rects[i][0], rects[i][1], rects[i][2], rects[i][3])
                face_frames.append(ff)

            # アニメーション、出力部分
            while True:
                print("animation start")
                ret, frame = cap.read()

                # 人数が変更したらまた位置から認識する
                x = len(rects)
                rects = []
                try:
                    for i in range(len(face_rect_manager)):
                        rect = []
                        # print("i", i)
                        rect.append(face_rect_manager[i][0])# top
                        rect.append(face_rect_manager[i][1]) # bottom
                        rect.append(face_rect_manager[i][2]) # left
                        rect.append(face_rect_manager[i][3]) # right
                        rects.append(rect)
                except:
                    print("something occured3")
                if x != len(rects):
                    print("len rects")
                    print("broke")
                    break

                # 鏡のように表示
                frame = cv2.flip(frame, 1)

                frame_manager[:] = []
                frame_manager.append(list(frame))

                # 以下画像加工部分(前半部分はCV２を用いている)
                # オーバーレイの作成
                overlay = frame.copy()
                # 距離データ枠の挿入
                if similar_distance_manager:
                    cv2.rectangle(overlay, (0, 320), (185, 40), (0, 0, 0), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

                # 距離データの挿入
                try:
                    for i in range(len(distance[0])):
                        distance[0][i] = distance[0][i] + random.uniform(0, 0.00000001)
                    cv2.putText(frame, "distances",(30, 75), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
                    for i in range(len(distance[0])):
                        d = round(distance[0][i], similar_windows[0][0].time%20)
                        cv2.putText(frame, str(d),(30, 100+25*i), font, 0.5,(255,255,255),2,cv2.FONT_HERSHEY_TRIPLEX)
                except:
                    pass

                # 顔認識フレーム情報の更新
                for i in range(len(face_frames)):
                    try:
                        face_frames[i].top = face_rect_manager[i][0]
                        face_frames[i].bottom = face_rect_manager[i][1]
                        face_frames[i].left = face_rect_manager[i][2]
                        face_frames[i].right = face_rect_manager[i][3]
                    except:
                        # 途中のアクションは継続する
                        pass

                # フレームデータのPIL変換
                frame = frame[:, :, ::-1].copy()
                frame = Image.fromarray(frame)

                # 顔認識のフレーム表示
                for i in range(len(face_frames)):
                    frame = face_frames[i].draw_frame(frame)

                face_frame_centers = []
                for i in range(len(face_frames)):
                    face_frame_centers.append(face_frames[i].cal_center())


                # 類似顔表示
                # 最初の10で出現、30待機、最後の20で消滅
                end_frame_num = 10
                wait_frame_num = 40
                easing_type = "ease_in_out_circular"
                for i in range(len(similar_windows)):
                    rect_top = rects[i][0]
                    rect_bottom = rects[i][1]
                    rect_left = rects[i][2]
                    rect_right = rects[i][3]
                    rect = (rect_left, rect_top, rect_right, rect_bottom)
                    face_frame_center_x = face_frame_centers[i][1]
                    face_frame_center_y = face_frame_centers[i][0]
                    for j in reversed(range(len(similar_windows[i]))):
                        t = similar_windows[i][j].time
                        if t < end_frame_num:
                            x = easing.easing(t, face_frame_center_x, similar_windows[i][j].movement_amount_x, end_frame_num, easing_type)
                            y = easing.easing(t, face_frame_center_y, similar_windows[i][j].movement_amount_y, end_frame_num, easing_type)
                        elif t < wait_frame_num:
                            t = end_frame_num
                            x = easing.easing(t, face_frame_center_x, similar_windows[i][j].movement_amount_x, end_frame_num, easing_type)
                            y = easing.easing(t, face_frame_center_y, similar_windows[i][j].movement_amount_y, end_frame_num, easing_type)
                        else:
                            x = easing.easing(t%wait_frame_num, face_frame_center_x+similar_windows[i][j].movement_amount_x, -similar_windows[i][j].movement_amount_x, end_frame_num, easing_type)
                            y = easing.easing(t%wait_frame_num, face_frame_center_y+similar_windows[i][j].movement_amount_y, -similar_windows[i][j].movement_amount_y,end_frame_num, easing_type)

                        # 情報更新・直線描画
                        lines[i][j].setter(face_frame_center_x, face_frame_center_y, x, y)
                        lines[i][j].draw_line(frame, rect)

                        # 欄外に%データの表示
                        try:
                            if j<3:
                                draw = ImageDraw.Draw(frame)
                                radius = similar_windows[i][j].get_radius()
                                font = ImageFont.truetype("arial.ttf", int(radius*(1/3)))
                                d = distance[0][j]
                                print("d:{}".format(d))
                                draw.text((int(x-radius*(2/3)+1), y+radius+1), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)+1), y+radius-1), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)+1), y+radius), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)-1), y+radius+1), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)-1), y+radius-1), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)-1), y+radius), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)), y+radius+1), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)), y+radius-1), str(round((1-d)*100, 3))+"%", font=font, fill=(0,0,0,128))
                                draw.text((int(x-radius*(2/3)), y+radius), str(round((1-d)*100, 3))+"%", font=font, fill=(255,255,255,128))
                        except:
                            print("something is occured in % phase")
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)

                        x = int(x)
                        y = int(y)
                        print("xy",x, y)
                        similar_windows[i][j].put_on_frame(frame=frame, place=[y, x])
                        print("ターゲット{}, {}番目の顔".format(i, j))

                # アニメーション終了から15フレーム後に次の検索に入る
                if similar_windows[0][0].time >= end_frame_num*2+wait_frame_num:
                    break


                # frame = frame.resize((frame.width*2, frame.height*2))
                # cv2への変換
                frame=np.asarray(frame)
                frame = frame[:, :, ::-1]

                cv2.imshow('FaceMandara', frame)
                k = cv2.waitKey(1)
            if k == 27:
                print("released!")
                break
        cap.release()
        cv2.destroyAllWindows()
        print("release camera!!!")
