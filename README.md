# Detección de fraudes por medio de machine learning

En este proyecto abordamos un problema real y costoso: el fraude en reclamos de seguros de autos. Nuestro objetivo es desarrollar un modelo de Machine Learning capaz de identificar automáticamente reclamos potencialmente fraudulentos, ayudando así a las aseguradoras a reducir pérdidas, optimizar procesos y proteger a los clientes legítimos.
Utilizamos técnicas de clasificación supervisada y procesamiento de datos para construir una solución eficiente, escalable y alineada con casos de uso reales en la industria aseguradora.

![image](https://github.com/user-attachments/assets/d6a09e13-142f-4679-9fb0-f922eb04ab71)

# El problema

El fraude en los seguros de autos representa un problema crítico para la industria. Solo en EE. UU., se estima que las aseguradoras pierden al menos $29 mil millones al año, según un estudio de Verisk (2017). Este tipo de fraude puede presentarse de distintas formas, como reclamos inflados, choques simulados o documentación falsa.

Estas prácticas no solo afectan las finanzas de las aseguradoras, sino también tienen un impacto directo en los usuarios legítimos: las primas aumentan y la rentabilidad general del sistema se reduce.

El mayor desafío es que la detección tradicional, manual, resulta costosa y lenta, especialmente frente a grandes volúmenes de datos. Por eso, es clave incorporar soluciones automáticas y escalables como el Machine Learning para enfrentar este problema.

![image](https://github.com/user-attachments/assets/2fb51e42-6b53-4525-ae3f-c030c2b95254)

# Nuestra solución

Para enfrentar el reto del fraude en reclamos de seguros, proponemos una solución basada en aprendizaje automático supervisado. Específicamente, implementamos un modelo de clasificación binaria que predice si un reclamo es fraudulento o no.

Este modelo aprende a partir de datos históricos previamente etiquetados (variable objetivo: FraudFound_P), lo que le permite identificar patrones comunes en los fraudes y aplicarlos a nuevos casos de forma automática.

Nuestro enfoque busca minimizar los falsos negativos, es decir, los fraudes que no son detectados, pero sin castigar a los usuarios legítimos. Así, logramos mantener un equilibrio entre precisión, eficiencia operativa y experiencia del cliente.

![image](https://github.com/user-attachments/assets/f31bcbed-abed-4c6f-927c-0cafba228d57)

# Metodología

# 1. Análisis exploratorio 
Comenzamos nuestro proyecto realizando un análisis exploratorio del conjunto de datos, el cual contiene 15,420 registros y 33 columna que describen distintos aspectos de cada reclamo.
Confirmamos que no existen valores nulos, lo cual facilita el procesamiento inicial. Las variables fueron categorizadas en distintos grupos para entender mejor su estructura y posibles relaciones con el fraude:

1 variable target (FraudFound_P)
9 temporales
4 demográficas 
2 geográficas 
5 vehiculares 
6 relacionadas a seguros/servicios
6 de circunstancias del accidente e historial

![image](https://github.com/user-attachments/assets/37d4929f-3acd-4040-ac3b-b48bd24e09f3)

Al analizar la distribución de la variable target, observamos un fuerte desbalance de clases:
94.01% de los reclamos están etiquetados como "No fraude
Solo 5.98% corresponden a casos "fraudulentos"
Esto significa que, aunque el dataset sea amplio, los ejemplos positivos (fraudes reales) son muy pocos. Este desbalance representa un reto importante en el modelado, ya que los algoritmos tienden a favorecer la clase mayoritaria.
Por esta razón, será necesario aplicar técnicas de balanceo de clases, para asegurar que el modelo aprenda a detectar correctamente los fraudes sin quedar sesgado.

![image](https://github.com/user-attachments/assets/bdb6f727-99a1-4179-89ce-df85eb82a8a0)

Al comparar variables del dataset con FraudFound_P, identificamos patrones interesantes:
Mes del reclamo (Month) vs Fraude:
 Enero destaca como el mes con mayor cantidad de reclamos fraudulentos, lo que sugiere una posible estacionalidad en el comportamiento del fraude.
Marca del vehículo (Make) vs Fraude:
 La marca Pontiac aparece con mayor frecuencia entre los casos de fraude. Esto podría indicar una correlación fuerte que afecte la predicción, por lo que se debe analizar si este peso es real o producto de un sesgo en los datos.

![image](https://github.com/user-attachments/assets/8012fcf2-ca4b-4401-ad06-e74cfa8636c2)


Continuamos analizando variables que pueden influir en la ocurrencia de fraudes:
Área del accidente (AccidentArea):
 La mayoría de los accidentes registrados en el dataset ocurrieron en zonas urbanas, lo que sugiere un sesgo hacia regiones más pobladas. Esto puede reflejar tanto la densidad vehicular como la frecuencia de reclamos.
Sexo del conductor (Sex):
 Observamos que la mayoría de los accidentes están relacionados con conductores hombres. Este tipo de patrón puede ser relevante al evaluar el riesgo, pero también debe manejarse con cuidado para evitar conclusiones sesgadas o discriminatorias en el modelo.

![image](https://github.com/user-attachments/assets/adde51b7-ec75-443d-8784-65012cc3a0e7)

En general, no se detecta multicolinealidad severa en el conjunto de datos: la mayoría de las correlaciones son bajas o cercanas a cero.
Correlaciones destacadas:
Existe una fuerte correlación positiva (0.94) entre PolicyNumber y Year, lo cual sugiere que podrían estar representando información redundante o relacionada temporalmente.

Una moderada correlación (0.28) entre WeekOfMonth y WeekOfMonthClaimed, lógica por la cercanía temporal entre la fecha de ocurrencia y la fecha de reclamo.
 Conclusión:
Variables como PolicyNumber y Year requieren un análisis más profundo para decidir si deben ser transformadas, combinadas o eliminadas, dependiendo de su valor predictivo y el riesgo de sobreajuste.

El resto de las variables numéricas no presenta problemas serios de correlación que puedan afectar negativamente el modelado.

![image](https://github.com/user-attachments/assets/76f9db69-3e29-48ef-b922-205f0963ffb5)

# 2. Limpieza de datos
Durante el proceso de limpieza de datos, se abordaron varios problemas para mejorar el rendimiento del modelo:
Se eliminaron los registros con valores faltantes en "Age" y se redujo la multicolinealidad para evitar su impacto negativo.
Las variables "PolicyNumber" y "RepNumber", al ser identificadoras, no aportan a la predicción, por lo que se eliminaron.
También se descartaron "Month" y "Year", ya que no son relevantes para la detección de fraude.

![image](https://github.com/user-attachments/assets/ea7ed4fb-caf5-48fc-86cd-664773e111be)


# 3. Separación del data set 

Para poder empezar a tomar desiciones con respecto a la modelación de los datos primero se debe partir el data set, esto con la finalidad de tomar 3 conjuntos de datos: el data set de entrenamiento, con el cual se harán las pruebas iniciales, el data set de validación, que permitirá evaluar las métricas del modelo, y finalmente el data set de test, con el cual se realizará la prueba final para identificar si el modelo es efectivo para la detección de fraudes. 

![image](https://github.com/user-attachments/assets/8950c5f9-243a-469c-a25e-72263f0a3141)

# 4. Creación de nuevas columnas
**notas feature engineering feature 1
![image](https://github.com/user-attachments/assets/2b569830-309a-4284-a9d9-e9c698a40c7d)

**notas feature engineering feature 2
![image](https://github.com/user-attachments/assets/95ea31ed-5511-4577-a20a-92aad272a3a1)

![image](https://github.com/user-attachments/assets/e7aca9f1-eee4-4ee3-95e3-6b4585b0b3d3)

# 4. Encoding
Se detectó el tipo de datos, columnas ordinales, no ordinales y binarias para escoger el mejor método de encoding.
![image](https://github.com/user-attachments/assets/c3d51193-d514-4b1d-906c-aa5581b62c9e)


# 5. Balanceo de la target
Para mejorar la detección de fraudes, se aplicó SMOTE al conjunto de entrenamiento, elevando los casos de fraude a aproximadamente el 30%. Esto ayudó a que el modelo reconociera mejor la clase minoritaria sin alterar demasiado la distribución original.
![image](https://github.com/user-attachments/assets/993fe080-48be-4021-869c-23fab33c8eb5)


# 6. Entrenamiento del modelo

En esta fase, se entrenó un modelo de ensamblado mediante votación que integra múltiples algoritmos de clasificación. Este enfoque permite al modelo aprender a partir del conjunto de entrenamiento, utilizando las variables seleccionadas como predictores y los registros de fraude como variable objetivo. La combinación de diferentes clasificadores mejora el desempeño general, ya que aprovecha las fortalezas individuales de cada modelo. Este esquema de ensamblado se utilizó con el propósito de identificar el modelo con el mejor rendimiento.

![image](https://github.com/user-attachments/assets/6bf19557-5c09-4a08-bd74-a707e6411ab1)

El modelo XGBClassifier mostró el mejor rendimiento, especialmente al identificar casos de fraude, con una alta precisión del 99%. Aunque aún no detecta todos los fraudes (recall menor al 50%), logró una mejora notable frente a los otros modelos. En general, es un modelo equilibrado y efectivo, por lo que se considera una buena opción para detectar fraudes.

![image](https://github.com/user-attachments/assets/6d4b8daa-f429-49c3-a095-3cb73527be74)


# 7.Tuneo de hiperparámetros

Se utilizó XGBoost y se hizo uso de Gridsearch para utilizar los mejores parámetros posibles al modelo.

![image](https://github.com/user-attachments/assets/6ba53618-22f0-4796-949f-eda7f86a5337)

# Explicabilidad
Utilizamos SHAP para interpretar el modelo y verificar que las decisiones fueran coherentes:
    Variables más importantes:
        bin__Fault (culpa en el accidente) fue el factor más determinante.
        cat__BasePolicy_Liability (tipo de póliza) también tuvo gran influencia.

    Análisis específico:
        Ejemplos con predicción de no fraude mostraron que un fuerte peso negativo en bin__Fault contrarrestaba otros factores de riesgo.
        El modelo no actúa como una caja negra: las predicciones tienen sentido de negocio y son interpretables, aumentando la confianza para su uso en producción.
![image](https://github.com/user-attachments/assets/d08af5ed-c06d-404a-8445-24b88f60e64a)

![image](https://github.com/user-attachments/assets/15913db5-8451-4736-9dea-2a0c808411a2)


# Desiciones de mejora 
Para mejorar el modelo, se crearon dos nuevas variables: ClienteMuyRiesgoso y PerfilRiesgoso, ambas binarias. Estas se activan únicamente si se cumplen tres condiciones específicas relacionadas con la responsabilidad del cliente, el tipo de póliza ("Liability") y la cantidad de reclamos pasados (exactamente tres). Además, se realizaron dos ajustes técnicos adicionales: se modificó el umbral de predicción a 0,5 y se trabajó en el ajuste de hiperparámetros para optimizar el desempeño del modelo.
![image](https://github.com/user-attachments/assets/8e7e842f-7e0e-4213-b919-161f60190788)


# Resultados
Tras ajustar el umbral de decisión para maximizar el F1-score, el modelo obtuvo:

    Recall en fraude: 61%
    → Se cumplió el objetivo de minimizar el fraude no detectado.

    Precision en fraude: 20%
    → Aunque baja, es aceptable considerando que se priorizó Recall.

    F1-Score en fraude: 30%
    → Mejor balance entre precisión y recall respecto a versiones anteriores.

    Accuracy general: 83%
    → Refleja un rendimiento global fuerte, adecuado para producción.
En conclusión se logró un equilibrio favorable entre capturar fraudes y minimizar falsos positivos, mejorando la eficiencia operativa.
![image](https://github.com/user-attachments/assets/878b669a-8c28-4872-a111-ce8d96a22171)
![image](https://github.com/user-attachments/assets/6d8aed76-0217-4d83-b1c9-d7d2f233135c)
![image](https://github.com/user-attachments/assets/111e873e-4172-4d40-ba90-34b0cb312494)

Impacto económico traducido:

    Fraudes no detectados (40 casos): $400,000 en pérdidas.
    Investigaciones innecesarias (310 casos): $310,000 en costos operativos.
    Costo total asociado: $710,000

![image](https://github.com/user-attachments/assets/e46cc328-84c1-46f2-b47a-763ce045b1b4)
Beneficio estratégico:

    Sin el modelo, habrían investigado 1450 reclamos manualmente.
    Con Machine Learning, solo investigan 310 reclamos sospechosos.
    Ahorro operativo estimado: más del 75%.
![image](https://github.com/user-attachments/assets/d8431ad0-6ecb-42af-a03f-e3a3f8cfa636)
![image](https://github.com/user-attachments/assets/61d1f642-8f6d-4c5e-9730-40454ba598d8)

# Implementación 
Flujo de aplicación:

    Ingreso de reclamo:
        Cliente presenta solicitud (accidente, robo, etc.)
    Predicción automática:
        El modelo evalúa en tiempo real la probabilidad de fraude.
    Clasificación:
        Alta probabilidad → investigación humana.
        Baja probabilidad → procesamiento automático normal.
    Acción de negocio:
        Casos riesgosos se auditan.
        Casos legítimos se procesan más rápido.
![image](https://github.com/user-attachments/assets/2fbf58d5-ec28-4ba0-8ebe-92c65b8744b1)
![image](https://github.com/user-attachments/assets/6346c9ab-c98d-4949-be6b-10949a50bd4e)

# Beneficios 
La implementación del modelo permite optimizar el uso de recursos humanos, enfocando a los investigadores en casos de mayor riesgo. Además, contribuye a reducir pérdidas interceptando más fraudes antes del pago y mejora los tiempos de respuesta al agilizar el trámite de reclamos legítimos mediante procesos automáticos. Esta estrategia está alineada con prácticas reales de aseguradoras como Allstate, Lemonade y Progressive, que integran inteligencia artificial en la detección de fraudes dentro de sus operaciones.

![image](https://github.com/user-attachments/assets/63324da4-db37-4323-94eb-6e4a4b884bc0)

# Limitaciones
Entre las principales limitaciones técnicas se identificó el desbalance de clases, ya que los casos de fraude representaban apenas un 6%, dificultando alcanzar alta precisión y recall al mismo tiempo. Para priorizar el recall, se sacrificó la precisión, lo que generó una alta cantidad de falsos positivos y aumentó la carga de trabajo manual. Además, no se exploraron algoritmos más avanzados que podrían haber mejorado el desempeño del modelo. A nivel de implementación, el principal desafío es el costo operativo asociado al volumen de investigaciones manuales, resaltando la necesidad de optimizar mejor el equilibrio entre recall y precisión.

![image](https://github.com/user-attachments/assets/20f05d20-453e-43e7-a6e3-7965a544813a)


# Conclusiones
Se alcanzó el objetivo principal al lograr un recall del 60,4% en la detección de fraudes, enfocándose en minimizar los casos no identificados. El modelo genera un valor tangible para el negocio, ya que la disminución de pérdidas supera los costos asociados a la investigación de falsos positivos. Aunque existen oportunidades de mejora, el modelo es suficientemente sólido para ser implementado como apoyo en el proceso de detección de fraudes en aseguradoras.

![image](https://github.com/user-attachments/assets/ae68edbe-908c-4c05-981d-c2101f0bccf0)

# Recomendaciones
Se recomienda probar modelos más sofisticados, como LightGBM, para mejorar el desempeño del sistema. También se sugiere optimizar el umbral de decisión con el fin de balancear precisión y recall según los objetivos del negocio. Adicionalmente, es importante definir y agregar nuevas variables relevantes, complementando la data actual con información como historial financiero o tipo de siniestro. Finalmente, se plantea evaluar los costos operativos reales para dimensionar adecuadamente el equipo de investigación de fraudes.

![image](https://github.com/user-attachments/assets/594330c8-b941-48ab-80c3-f8ceb0012c72)

# Future Work
Se propone explorar modelos de machine learning más avanzados, incluyendo algoritmos como XGBoost, LightGBM y CatBoost, con el objetivo de mejorar tanto el recall como la precisión del modelo. Además, se plantea realizar un ajuste fino del umbral de decisión para optimizar el equilibrio entre recall y precisión, alineándolo con el nivel de riesgo de fraude que la empresa esté dispuesta a asumir. Finalmente, se sugiere incrementar la cantidad de datos disponibles para fortalecer el entrenamiento y la capacidad predictiva del modelo.

![image](https://github.com/user-attachments/assets/2a1def57-3bd8-4105-8ab5-5561f66f70c6)

Gracias!
![image](https://github.com/user-attachments/assets/05d8cea3-eb59-4695-b091-ccc913c1eead)


