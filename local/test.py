import os
os.chdir(r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local')

from detector_multiples_hojas_yolo import detectar_y_clasificar_hojas

# Cambia esta ruta por la de tu imagen de prueba
resultado = detectar_y_clasificar_hojas(
    r'C:\Users\User\Documents\GitHub\consumidor_api_cacao\local\dataset_cacao\Potasio\test_v1.jpeg',
    #filtrar_solo_plantas=False,  # Desactivamos el filtro para la prueba
    mostrar_resultados=True
)