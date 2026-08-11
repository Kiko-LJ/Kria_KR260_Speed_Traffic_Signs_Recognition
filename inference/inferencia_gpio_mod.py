import os
import cv2
import numpy as np
import xir
import vart
import time
import mmap
import sys
from threading import Thread, Lock

# ==============================================================================
# CONFIGURACIÓN DEL MODELO YOLOv5 (ACTUALIZADO A 10 CLASES)
# ==============================================================================
CLASS_NAMES = [
    "Velocidad 100", 
    "Velocidad 120", 
    "Velocidad 20", 
    "Velocidad 30", 
    "Velocidad 40", 
    "Velocidad 50", 
    "Velocidad 60", 
    "Velocidad 70", 
    "Velocidad 80", 
    "Velocidad 90"
]

ANCHORS_P3 = [[10, 13], [16, 30], [33, 23]]       
ANCHORS_P4 = [[30, 61], [62, 45], [59, 119]]      
ANCHORS_P5 = [[116, 90], [156, 198], [373, 326]]  

CONF_THRESHOLD = 0.35  
IOU_THRESHOLD = 0.45
LOGIT_THRESHOLD = -np.log(1.0 / CONF_THRESHOLD - 1.0)


# ==============================================================================
# CONTROLADOR HARDWARE DEL INTERRUPTOR (GPIO SYSFS)
# ==============================================================================
class InterruptorGPIO:
    # Clase para leer el estado del pin GPIO 499
    def __init__(self, pin="499"):
        self.pin = pin
        self.path = f"/sys/class/gpio/gpio{self.pin}"
        self._setup()

    def _setup(self):
        # 1. Exportar el pin si no está expuesto en el sistema de archivos de Linux
        if not os.path.exists(self.path):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(self.pin)
                time.sleep(0.1) # Breve pausa para asegurar la creación del nodo
            except IOError:
                pass # Ya exportado previamente o sin permisos temporales

        # 2. Configurar siempre la dirección como entrada ("in")
        try:
            with open(f"{self.path}/direction", "w") as f:
                f.write("in")
        except IOError:
            print(f"Error al acceder al GPIO {self.pin}. Asegura permisos de administrador.")

    def leer_estado(self):
        # Devuelve 1 si el interruptor está a 3.3V, y 0 si está a tierra (GND)
        try:
            with open(f"{self.path}/value", "r") as f:
                return int(f.read().strip())
        except IOError:
            # En caso de fallo de lectura, asumimos por seguridad que está ACTIVO para no bloquear el programa
            return 1


# ==============================================================================
# HILO ASÍNCRONO DE CAPTURA DE CÁMARA
# ==============================================================================
class AsyncCamera:
    def __init__(self, cam_index=0):
        os.system(f"v4l2-ctl -d /dev/video{cam_index} --set-fmt-video=width=640,height=480,pixelformat=MJPG 2>/dev/null")
        os.system(f"v4l2-ctl -d /dev/video{cam_index} -c exposure_auto_priority=0 2>/dev/null")
        
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = Lock()

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is None:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()


# ==============================================================================
# DECODIFICADOR VECTORIAL HÍBRIDO
# ==============================================================================
def decode_yolov5_layer_fast(tensor_data, scale, anchors, net_w, net_h, int_raw_logit_thresh, conf_thresh):
    grid_h, grid_w = tensor_data.shape[1], tensor_data.shape[2]
    raw_data = tensor_data[0].reshape((grid_h, grid_w, 3, 5 + len(CLASS_NAMES)))
    
    # 1. Filtrado en formato INT8
    candidate_mask = raw_data[..., 4] > int_raw_logit_thresh
    if not np.any(candidate_mask):
        return []
        
    cy_indices, cx_indices, a_indices = np.where(candidate_mask)
    cand_data_raw = raw_data[candidate_mask]
    
    # 2. Argmax sobre INT8
    raw_class_logits = cand_data_raw[:, 5:]
    class_ids = np.argmax(raw_class_logits, axis=1)
    winning_raw_logits = raw_class_logits[np.arange(len(class_ids)), class_ids]
    
    # 3. Escalado flotante exclusivo para los supervivientes iniciales
    obj_logits = cand_data_raw[:, 4].astype(np.float32) * scale
    class_logits = winning_raw_logits.astype(np.float32) * scale
    
    # 4. Cálculo de confianza reducida
    obj_conf = 1.0 / (1.0 + np.exp(-np.clip(obj_logits, -20, 20)))
    class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_logits, -20, 20)))
    final_scores = class_scores * obj_conf
    
    # 5. Segundo filtro estricto ANTES de decodificar geometría
    final_mask = final_scores > conf_thresh
    if not np.any(final_mask):
        return []
        
    cand_data_raw = cand_data_raw[final_mask]
    final_scores = final_scores[final_mask]
    class_ids = class_ids[final_mask]
    cy_indices = cy_indices[final_mask]
    cx_indices = cx_indices[final_mask]
    a_indices = a_indices[final_mask]
    
    # 6. Decodificación geométrica SÓLO para los objetos reales detectados
    cand_data_float = cand_data_raw[:, :4].astype(np.float32) * scale
    tx, ty, tw, th = cand_data_float[:, 0], cand_data_float[:, 1], cand_data_float[:, 2], cand_data_float[:, 3]
    
    sx = 1.0 / (1.0 + np.exp(-np.clip(tx, -20, 20)))
    sy = 1.0 / (1.0 + np.exp(-np.clip(ty, -20, 20)))
    sw = 1.0 / (1.0 + np.exp(-np.clip(tw, -20, 20)))
    sh = 1.0 / (1.0 + np.exp(-np.clip(th, -20, 20)))
    
    bx = (cx_indices + 2.0 * sx - 0.5) / grid_w
    by = (cy_indices + 2.0 * sy - 0.5) / grid_h
    
    anchors_arr = np.array(anchors)
    cand_anchors = anchors_arr[a_indices]
    bw = (cand_anchors[:, 0] * (2.0 * sw) ** 2) / net_w
    bh = (cand_anchors[:, 1] * (2.0 * sh) ** 2) / net_h
    
    xmin = np.maximum(0, (bx - bw / 2.0) * net_w).astype(np.int32)
    ymin = np.maximum(0, (by - bh / 2.0) * net_h).astype(np.int32)
    w_box = np.minimum(net_w - xmin, (bw * net_w)).astype(np.int32)
    h_box = np.minimum(net_h - ymin, (bh * net_h)).astype(np.int32)
    
    return [[int(xmin[i]), int(ymin[i]), int(w_box[i]), int(h_box[i]), float(final_scores[i]), int(class_ids[i])] for i in range(len(xmin))]


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def main():
    model_path = "yolov5_kria_FINAL.xmodel"
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if not os.path.exists(model_path):
        print(f"Error: Falta el modelo {model_path}")
        return

    print("Cargando modelo en la DPU...")
    graph = xir.Graph.deserialize(model_path)
    root_subgraph = graph.get_root_subgraph()
    dpu_subgraph = [s for s in root_subgraph.get_children() if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
    runner = vart.Runner.create_runner(dpu_subgraph, "run")

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_shape = input_tensors[0].dims
    net_h, net_w = input_shape[1], input_shape[2]

    # Inicializar el lector del interruptor físico (GPIO 499)
    print("Vinculando Switch de Hardware (Pin GPIO 499)...")
    switch = InterruptorGPIO(pin="499")

    print(f"Inicializando Cámara Asíncrona en /dev/video{cam_index}...")
    cam = AsyncCamera(cam_index).start()

    # Mapeo de pantalla
    with open("/sys/class/graphics/fb0/virtual_size", "r") as f:
        fb_w, fb_h = map(int, f.read().strip().split(','))
    
    fb_dev = os.open("/dev/fb0", os.O_RDWR)
    fb_mem = mmap.mmap(fb_dev, fb_w * fb_h * 2, mmap.MAP_SHARED, mmap.PROT_WRITE)
    fb_array = np.ndarray(shape=(fb_h, fb_w, 2), dtype=np.uint8, buffer=fb_mem)
    fb_array.fill(0)

    input_data = [np.empty(input_shape, dtype=np.int8)]
    output_data = [np.empty(tensor.dims, dtype=np.int8) for tensor in output_tensors]
    
    input_scale = 2 ** input_tensors[0].get_attr("fix_point")
    combined_input_scale = input_scale / 255.0

    # OPTIMIZACIÓN ENTRADA: Creación de Look-Up Table estática
    input_lut = np.clip(np.arange(256) * combined_input_scale, -128, 127).astype(np.int8)

    frame_count = 0
    t_cam_total, t_dpu_total, t_post_total, t_render_total = 0, 0, 0, 0

    print("Ejecutando tubería híbrida con control de Switch por hardware...")

    try:
        while True:
            t_start = time.time()
            
            # 1. Captura instantánea desde memoria compartida
            ret, frame_raw = cam.read()
            t_after_cam = time.time()
            if not ret or frame_raw is None: 
                continue
                
            frame = frame_raw.copy() 

            # 2. Comprobar estado del Interruptor Físico
            modo_inferencia = (switch.leer_estado() == 1)

            if modo_inferencia:
                # ==============================================================
                # MODO ACTIVO: INFERENCIA DE ALTO RENDIMIENTO DPU
                # ==============================================================
                img_resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # Mapeo indexado
                input_data[0][0] = input_lut[img_rgb]

                t_before_dpu = time.time()
                job_id = runner.execute_async(input_data, output_data)
                runner.wait(job_id)
                t_after_dpu = time.time()

                # 3. Postprocesamiento inteligente
                all_boxes = []
                for i, tensor in enumerate(output_tensors):
                    fixpos = tensor.get_attr("fix_point")
                    output_scale = 2 ** (-fixpos)
                    grid_w = tensor.dims[2]
                    
                    if grid_w == 80: anchors = ANCHORS_P3
                    elif grid_w == 40: anchors = ANCHORS_P4
                    elif grid_w == 20: anchors = ANCHORS_P5
                    else: continue
                    
                    # Precalculamos el umbral logit entero nativo para la capa
                    raw_logit_thresh = LOGIT_THRESHOLD / output_scale
                    int_raw_logit_thresh = int(np.ceil(raw_logit_thresh))
                        
                    layer_boxes = decode_yolov5_layer_fast(output_data[i], output_scale, anchors, net_w, net_h, int_raw_logit_thresh, CONF_THRESHOLD)
                    all_boxes.extend(layer_boxes)

                final_boxes = []
                if len(all_boxes) > 0:
                    nms_boxes = [b[0:4] for b in all_boxes]
                    nms_scores = [b[4] for b in all_boxes]
                    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, CONF_THRESHOLD, IOU_THRESHOLD)
                    if len(indices) > 0:
                        for idx in indices.flatten():
                            final_boxes.append(all_boxes[idx])

                scale_x = frame.shape[1] / net_w
                scale_y = frame.shape[0] / net_h

                for box in final_boxes:
                    xmin, ymin, w_box, h_box, score, class_id = box
                    abs_xmin = int(xmin * scale_x)
                    abs_ymin = int(ymin * scale_y)
                    abs_xmax = int((xmin + w_box) * scale_x)
                    abs_ymax = int((ymin + h_box) * scale_y)
                    
                    label = f"{CLASS_NAMES[class_id]} {score*100:.0f}%"
                    cv2.rectangle(frame, (abs_xmin, abs_ymin), (abs_xmax, abs_ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (abs_xmin, abs_ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                t_after_post = time.time()
                
                # Acumulación de tiempos de procesamiento
                t_cam_total += (t_after_cam - t_start)
                t_dpu_total += (t_after_dpu - t_before_dpu)
                t_post_total += (t_after_post - t_after_dpu)

            else:
                # ==============================================================
                # MODO INACTIVO (BYPASS): SÓLO CÁMARA (AHORRO ENERGÍA)
                # ==============================================================
                # Dibujamos un marco indicador de pausa y texto de aviso en rojo
                cv2.rectangle(frame, (0, 0), (639, 479), (0, 0, 255), 4)
                cv2.putText(frame, "INFERENCIA: DESACTIVADA (SW OFF)", (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                t_after_post = time.time()
                
                # Sumamos la latencia de captura, pero ponemos a cero los tiempos de DPU y Postproceso
                t_cam_total += (t_after_cam - t_start)
                t_dpu_total += 0
                t_post_total += 0

            # 4. Volcado instantáneo a pantalla (640x480 sin reescalar)
            frame_565 = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
            fb_array[0:480, 0:640, :] = frame_565
            
            t_after_render = time.time()
            t_render_total += (t_after_render - t_after_post)
            frame_count += 1

            # Imprimir métricas adaptadas al modo de ejecución
            if frame_count >= 15:
                print("\n[Metricas]")
                print(f"  Latencia Memoria Cámara:     {t_cam_total/15*1000:.2f} ms")
                if t_dpu_total > 0:
                    print(f"  Tiempo de Inferencia DPU:   {t_dpu_total/15*1000:.2f} ms")
                    print(f"  Tiempo Postproceso CPU:      {t_post_total/15*1000:.2f} ms")
                    print(f"  FPS DE EJECUCIÓN TOTAL:    {1.0 / ((t_cam_total+t_dpu_total+t_post_total+t_render_total)/15):.1f}")
                else:
                    print(f"  Tiempo de Inferencia DPU:   [PAUSADO]")
                    print(f"  Tiempo Postproceso CPU:      [PAUSADO]")
                    print(f"  FPS DE EJECUCIÓN TOTAL:    {1.0 / ((t_cam_total+t_render_total)/15):.1f} (Modo Pasivo)")
                print(f"  Tiempo Renderizado Pantalla:        {t_render_total/15*1000:.2f} ms")
                t_cam_total, t_dpu_total, t_post_total, t_render_total = 0, 0, 0, 0
                frame_count = 0

    except KeyboardInterrupt:
        print("\nDeteniendo de forma segura...")
    finally:
        cam.release()
        fb_mem.close()
        os.close(fb_dev)

if __name__ == "__main__":
    main()
