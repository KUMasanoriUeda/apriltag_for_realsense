import cv2
import pyrealsense2 as rs
import numpy as np
import sys
from argparse import ArgumentParser

####################################
#-------------事前準備-------------#
####################################

# AprilTagの環境パスを追加
ApriltagPath = "/home/robotlab/AprilTag/scripts"
sys.path.append(ApriltagPath)
import apriltag

# cv2の画面サイズ
WIDTH = 640
HEIGHT = 480

# ストリーミング初期化
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
frames = pipeline.wait_for_frames()

####################################
# カメラの内部パラメータを取得
intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
# 光学中心
fx = intrinsics.fx
fy = intrinsics.fy
# 焦点距離
cx = intrinsics.ppx
cy = intrinsics.ppy
####################################

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)
threshold = (WIDTH * HEIGHT * 3) * 0.95

# AprilTagの設定
parser = ArgumentParser(description='Detect AprilTags from streaming.')
apriltag.add_arguments(parser)
options = parser.parse_args()

# AprilTag検出器の初期化
detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

####################################
#----------ストリーミング----------#
####################################

while (True):
    # フレーム待ち
    frames = pipeline.wait_for_frames()

    # フレームの色情報を取得
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()

    # フレームの色情報をnumpy配列に変換
    color_image = np.asanyarray(color_frame.get_data())

    # AprilTagの検出
    result, tag_poses = apriltag.detect_tags(color_image, detector, [fx, fy, cx, cy], 0.03, 2, 3, False)

    # タグの位置を表示
    cv2.imshow("Apriltags", tag_poses)

    # aを押すと終了
    if cv2.waitKey(1) & 0xff == ord("a"):
        cv2.destroyAllWindows()
        break

# 後処理
pipeline.stop()
cv2.destroyAllWindows()

