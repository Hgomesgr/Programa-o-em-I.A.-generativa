import math
import time
import cv2
import numpy as np
import pyautogui
import streamlit as st
from ultralytics import YOLO

# Configurações de segurança e desempenho do PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

st.set_page_config(
    page_title="Controle do Mouse com Clique por Gestos",
    page_icon="🖐️",
    layout="wide"
)

@st.cache_resource
def load_yolo_pose(model_path: str = "yolov8n-pose.pt"):
    """Carrega o modelo YOLOv8 Pose em memória."""
    model = YOLO(model_path)
    return model

def main():
    st.title("🖐️ Controle do Cursor + Clique Direito (Abrir/Fechar Mão)")
    st.write("Mova a mão para guiá-lo e **feche e abra a mão** para acionar o **Clique com o Botão Direito** do mouse.")

    screen_w, screen_h = pyautogui.size()

    with st.spinner("Carregando modelo YOLOv8 Pose..."):
        model = load_yolo_pose("yolov8n-pose.pt")

    # Barra Lateral
    st.sidebar.header("⚙️ Configurações do Controle")
    camera_idx = st.sidebar.number_input("Índice da Câmera", min_value=0, max_value=5, value=0)
    conf_thresh = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.45, 0.05)
    
    smooth_factor = st.sidebar.slider(
        "Suavização do Cursor (EMA)",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05
    )

    gesture_sens = st.sidebar.slider(
        "Sensibilidade de Fechamento da Mão",
        min_value=20, max_value=100, value=45, step=5,
        help="Ajuste o limite de proximidade para identificar a mão fechada."
    )

    mão_escolhida = st.sidebar.radio(
        "Mão Principal",
        options=["Direita (Pulso Direito)", "Esquerda (Pulso Esquerdo)"],
        index=0
    )
    
    # Keypoints COCO: 9 = Pulso Esquerdo, 10 = Pulso Direito, 7 = Cotovelo Esquerdo, 8 = Cotovelo Direito
    target_wrist_idx = 10 if "Direita" in mão_escolhida else 9
    target_elbow_idx = 8 if "Direita" in mão_escolhida else 7

    run_camera = st.sidebar.toggle("1. Ligar Câmera", value=False)
    enable_mouse = st.sidebar.toggle("2. Ativar Controle do Mouse + Clique", value=False)

    col_video, col_info = st.columns([3, 1])

    with col_video:
        video_feed = st.image([])

    with col_info:
        st.subheader("📌 Painel de Controle")
        status_box = st.empty()
        gesture_box = st.empty()
        coords_box = st.empty()

    # Variáveis de Estado (Suavização e Trava do Clique)
    prev_x, prev_y = None, None
    hand_was_closed = False  # Trava para acionar o clique apenas 1 vez por fechamento

    if run_camera:
        cap = cv2.VideoCapture(int(camera_idx))

        if not cap.isOpened():
            st.error(f"Erro ao abrir a câmera no índice {camera_idx}.")
            return

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Falha ao capturar o frame.")
                break

            # Espelhar para movimentação intuitiva
            frame = cv2.flip(frame, 1)
            h_frame, w_frame, _ = frame.shape

            results = model.predict(
                source=frame,
                conf=conf_thresh,
                device="cpu",
                verbose=False
            )

            result = results[0]
            detected = False

            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints = result.keypoints.data.cpu().numpy()

                for person_kps in keypoints:
                    if len(person_kps) > target_wrist_idx:
                        wrist = person_kps[target_wrist_idx]
                        elbow = person_kps[target_elbow_idx]

                        wx, wy, wconf = wrist[0], wrist[1], wrist[2]
                        ex, ey, econf = elbow[0], elbow[1], elbow[2]

                        if wconf > conf_thresh:
                            detected = True

                            # 1. Mapeamento e Suavização do Mouse
                            target_x = np.interp(wx, [0, w_frame], [0, screen_w])
                            target_y = np.interp(wy, [0, h_frame], [0, screen_h])

                            if prev_x is None or prev_y is None:
                                curr_x, curr_y = target_x, target_y
                            else:
                                curr_x = prev_x + smooth_factor * (target_x - prev_x)
                                curr_y = prev_y + smooth_factor * (target_y - prev_y)

                            prev_x, prev_y = curr_x, curr_y

                            # 2. Detecção do Gesto (Distância entre Pulso e Cotovelo/Variação do Membro)
                            # Quando a mão fecha e encolhe em direção ao corpo, a distância relativa diminui
                            dist_wrist_elbow = math.hypot(wx - ex, wy - ey) if econf > conf_thresh else gesture_sens + 10
                            
                            is_hand_closed = dist_wrist_elbow < gesture_sens

                            # Lógica do Clique com Botão Direito (Disparo Único com Trava/Debounce)
                            click_event = False
                            if is_hand_closed and not hand_was_closed:
                                hand_was_closed = True
                                click_event = True
                                if enable_mouse:
                                    pyautogui.rightClick(int(curr_x), int(curr_y))
                            elif not is_hand_closed:
                                hand_was_closed = False  # Reseta o estado quando a mão abre novamente

                            # Mover o ponteiro se habilitado
                            if enable_mouse:
                                pyautogui.moveTo(int(curr_x), int(curr_y))
                                status_box.success("🖱️ Controle de Mouse Ativo")
                            else:
                                status_box.warning("⚠️ Ative 'Ativar Controle do Mouse + Clique' no menu lateral.")

                            # Visualização gráfica na tela da câmera
                            circle_color = (0, 0, 255) if is_hand_closed else (0, 255, 0) # Vermelho se fechada, Verde se aberta
                            cv2.circle(frame, (int(wx), int(wy)), 15, circle_color, -1)
                            
                            estado_texto = "MAO FECHADA (Clique Direito!)" if is_hand_closed else "MAO ABERTA"
                            cv2.putText(
                                frame, estado_texto, (int(wx) - 60, int(wy) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2
                            )

                            # Exibição dos painéis informativos
                            if click_event:
                                gesture_box.error("💥 **CLIQUE DIREITO EXECUTADO!**")
                            else:
                                if is_hand_closed:
                                    gesture_box.warning("✊ Mão Fechada")
                                else:
                                    gesture_box.info("🖐️ Mão Aberta")

                            coords_box.markdown(
                                f"**Posição do Mouse:** `({int(curr_x)}, {int(curr_y)})`\n\n"
                                f"**Distância do Membro:** `{int(dist_wrist_elbow)}` (Limiar: `{gesture_sens}`)"
                            )
                            break

            if not detected:
                status_box.info("Aguardando mão na câmera...")
                gesture_box.empty()
                coords_box.empty()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_feed.image(frame_rgb, use_container_width=True)

        cap.release()
    else:
        video_feed.empty()
        status_box.info("Câmera desligada. Marque '1. Ligar Câmera' para iniciar.")

if __name__ == "__main__":
    main()