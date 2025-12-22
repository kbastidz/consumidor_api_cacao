import os
os.chdir(r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local')

from detector_multiples_hojas_yolo_v_roboflow import detectar_y_clasificar_hojas

# Prueba con una imagen
resultado = detectar_y_clasificar_hojas(
    r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\dataset_cacao\Potasio\test_v11.jpeg',
    #filtrar_solo_plantas=False,
    mostrar_resultados=True
)