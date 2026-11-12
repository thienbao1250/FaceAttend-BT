
import cv2


def main():
    # Mở webcam mặc định
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Khong the mo camera!")
        return

    print("Camera da khoi dong!")
    print("Nhan Q de thoat.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Khong doc duoc frame!")
            break

        cv2.imshow("FaceAttend AI - Camera Test", frame)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
