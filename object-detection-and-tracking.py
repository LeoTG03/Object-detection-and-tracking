import numpy as np
import cv2 as cv
import argparse as arg

def read_image(path, resize=(400, 200)):
    """
    Lee una imagen de un archivo y la redimensiona.

    Args:
        path (str): Ruta del archivo de imagen.
        resize (tuple, optional): Dimensiones de redimensionado. Por defecto es (400, 200).

    Returns:
        numpy.ndarray: Imagen leída y redimensionada.
    """
    img = cv.imread(path)
    if img is None:
        print("Image Not Available")
        return None
    # Redimensionar la imagen a las dimensiones especificadas
    img_resized = cv.resize(img, resize)
    return img_resized


def read_parser():
    """
    Lee los argumentos de la línea de comandos.

    Returns:
        argparse.Namespace: Argumentos de la línea de comandos parseados.
    """
    parser = arg.ArgumentParser(description="Program for feature detection and matching with BRIEF descriptor using FAST detector")
    parser.add_argument("--img_obj", 
                        dest="image_object", 
                        type=str, 
                        help="Path to query image")
    parser.add_argument("--video", 
                        dest="video_path", 
                        type=str, 
                        help="Path to video sequence")
    args = parser.parse_args()
    return args

def brief_descriptor(img):
    """
    Extrae descriptores BRIEF de una imagen utilizando el detector FAST.

    Args:
        img (numpy.ndarray): Imagen de la que se extraerán los descriptores.

    Returns:
        tuple: Keypoints detectados y descriptores BRIEF.
    """
    fast = cv.FastFeatureDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp = fast.detect(img, None)
    kp, des = brief.compute(img, kp)

    if des is None:
        des = np.empty((0, 64))
    else:
        des = np.float32(des)

    return kp, des

def match_images(img_obj, video_path):
    """
    Encuentra coincidencias de características entre una imagen de consulta y un video.

    Args:
        img_obj (numpy.ndarray): Imagen de consulta.
        video_path (str): Ruta del video.

    Returns:
        None
    """
    kp1, des1 = brief_descriptor(img_obj)
    des1 = np.float32(des1)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Inicializar el estado anterior del cruce y contadores de cruces
    prev_crossing = None
    left_to_right_count = 0
    right_to_left_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        kp2, des2 = brief_descriptor(frame)
        des2 = np.float32(des2)

        flann = cv.FlannBasedMatcher()
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:  # Considerar solo cuando hay suficientes coincidencias
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calcular la caja delimitadora alrededor de los keypoints coincidentes
            min_x, min_y = np.min(dst_pts, axis=0)[0]
            max_x, max_y = np.max(dst_pts, axis=0)[0]

            # Calcular el punto medio de la caja delimitadora
            center_x = int((min_x + max_x) / 2)
            center_y = int((min_y + max_y) / 2)

            # Dibujar la caja delimitadora en el frame
            cv.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 255, 255), 2)

            # Dibujar el centro en el frame
            cv.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

            # Dibujar la línea en la mitad de la pantalla
            screen_middle = frame.shape[1] // 2
            cv.line(frame, (screen_middle, 0), (screen_middle, frame.shape[0]), (0, 255, 255), 2)

            # Verificar si el centro cruza la línea de la mitad de la pantalla
            if prev_crossing is not None:
                if prev_crossing < screen_middle <= center_x:
                    left_to_right_count += 1
                elif prev_crossing > screen_middle >= center_x:
                    right_to_left_count += 1

            # Actualizar el estado anterior del cruce
            prev_crossing = center_x

        # Dibujar los matches en el frame
        img_matches = cv.drawMatches(img_obj, kp1, frame, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Mostrar conteo de cruces en pantalla
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_matches, f"Izq a Der: {left_to_right_count}", (10, 30), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(img_matches, f"Der a Izq: {right_to_left_count}", (10, 60), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow("Matches", img_matches)
        if cv.waitKey(30) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def main():
    """
    Función principal que lee los argumentos de la línea de comandos, carga la imagen de consulta,
    y realiza el seguimiento de objetos en el video.
    """
    args = read_parser()
    img_obj = read_image(args.image_object)
    if img_obj is None:
        return

    match_images(img_obj, args.video_path)

if __name__ == "__main__":
    main()
