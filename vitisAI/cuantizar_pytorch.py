import os
import sys
import cv2
import numpy as np
import torch

# =====================================================================
# Parche para pesos de Google Colab (NumPy 2.x -> 1.x)
# =====================================================================
import numpy.core
sys.modules['numpy._core'] = numpy.core
try:
    import numpy.core.multiarray
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except ImportError:
    pass
# =====================================================================

# Añadir la carpeta de YOLOv5 al sistema para que PyTorch encuentre sus módulos internos
sys.path.append(os.path.abspath('./yolov5'))

# Importar el inicializador de cuantización nativo de Xilinx Vitis AI
from pytorch_nndct.apis import torch_quantizer

def cargar_imagenes_calibracion(ruta_imagenes, tamaño=(640, 640), limite=100):
    """Lee, procesa y normaliza las imágenes exactamente como lo hace YOLOv5"""
    if not os.path.exists(ruta_imagenes):
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {ruta_imagenes}")
        
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp')
    archivos = [os.path.join(ruta_imagenes, f) for f in os.listdir(ruta_imagenes) if f.lower().endswith(formatos_validos)]
    
    if len(archivos) == 0:
        raise ValueError(f"No se encontraron imágenes válidas en '{ruta_imagenes}'")
        
    print(f"Total de imágenes encontradas: {len(archivos)}.")
    print(f"Procesando las primeras {min(limite, len(archivos))} imágenes para la calibración INT8")
    
    tensores_imagenes = []
    for p in archivos[:limite]:
        img = cv2.imread(p)
        if img is None:
            continue
        # Preprocesamiento oficial de YOLOv5: BGR a RGB, redimensionar y normalizar a [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tamaño, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # Pasar de formato HWC a CHW
        tensores_imagenes.append(img)
        
    return np.array(tensores_imagenes)

def main():
    # Definición de las rutas locales según tu estructura organizada
    #archivo_pesos = 'best.pt'
    #archivo_pesos = 'best_new.pt'
    #archivo_pesos = 'best_fp32.pt'
    archivo_pesos = 'bestF.pt'
    #carpeta_calib = 'calib_images'
    carpeta_calib = 'calib_images_final_cut'
    carpeta_salida = 'output_vitis_final'
    
    # Forzar el uso de la CPU dentro del contenedor Docker estándar
    device = torch.device("cpu") 
    
    # Cargar el modelo desde tu archivo .pt
    print("Cargando estructura y pesos YOLOv5")
    try:
        checkpoint = torch.load(archivo_pesos, map_location=device)
        # Extraer el modelo flotante (PyTorch utilizará automáticamente nuestro yolo.py modificado)
        model = checkpoint['model'].float().eval()
        print("Modelo cargado correctamente en memoria.")
    except Exception as e:
        print(f"Error crítico al cargar los pesos: {e}")
        return

    # Preparar el lote de imágenes de calibración
    try:
        dataset_calib = cargar_imagenes_calibracion(carpeta_calib, limite=100)
    except Exception as e:
        print(e)
        return

    # Definir el tamaño de entrada fijo que espera la red (Batch: 1, Canales: 3, 640x640)
    tensor_entrada = torch.randn([1, 3, 640, 640]).to(device)

    # 4. PASADA 1: FASE DE CALIBRACIÓN 
    print("\n[Fase 1/2] Iniciando Calibración INT8 (Modo 'calib')")
    
    # Inicializar el cuantizador de Xilinx en modo calibración
    quantizer = torch_quantizer('calib', model, (tensor_entrada,), device=device)
    quant_model = quantizer.quant_model

    # Alimentar la red con imágenes reales para buscar los rangos de escala óptimos
    with torch.no_grad():
        for i, img_np in enumerate(dataset_calib):
            input_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            quant_model(input_tensor)
            if (i + 1) % 10 == 0:
                print(f" {i + 1}/{len(dataset_calib)} imágenes calibradas")

    # Guardar los mapas de cuantización temporales generados
    quantizer.export_quant_config()
    
    print("Fase de calibración finalizada con éxito.")

    # PASADA 2: EXPORTACIÓN DE LA ESTRUCTURA COMPILABLE
    print("\n[Fase 2/2] Generando modelo intermedio (.xmodel) (Modo 'test')...")
    # Reinicializamos en modo test para fijar los valores INT8 calculados

# Recargar el modelo fresco para limpiar las cuadrículas internas
    model = torch.load(archivo_pesos, map_location=device)['model'].float().eval()

    quantizer = torch_quantizer('test', model, (tensor_entrada,), device=device)
    quant_model = quantizer.quant_model

    with torch.no_grad():
        quant_model(tensor_entrada)

    # Exportar el archivo definitivo
    quantizer.export_xmodel(deploy_check=False, output_dir=carpeta_salida)
    
    print("\n¡PROCESO FINALIZADO CON ÉXITO!")
    print(f"Tu modelo intermedio se ha guardado en: {os.path.abspath(carpeta_salida)}")
    print("Siguiente paso: Ejecuta el comando de consola 'vai_c_xir' para compilarlo.")

if __name__ == "__main__":
    main()
