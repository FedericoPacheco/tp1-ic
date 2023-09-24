## Inteligencia computacional - Trabajo práctico N1
> Grupo 1

Integrantes:
- [Albino, Sebastián](https://github.com/Sebastian-Albino)
- [Pacheco Pilan, Federico](https://github.com/FedericoPacheco)
- [Rodriguez, Alejandro](https://github.com/alerodriguez01)

---

### Resumen
El presente trabajo práctico pretende emular el paper [*Aprobación del presidente de Perú basado en análisis de sentimientos de twitter*](https://journals.eagora.org/revTECHNO/article/view/4396) de *Luis Fernando Solis Navarro* de la *Universidad Nacional de San Cristóbal de Huamanga*, publicado el 28/12/2022 en la *Revista Internacional de Tecnología, Ciencia y Sociedad*. Este evalúa distintas configuraciones de redes neuronales para realizar análisis de sentimientos y lograr estimar la aprobación popular del presidente del Perú utilizando datos de Twitter.

Se implementaron cuatro modelos:
- Perceptrón multicapa con una capa de embedding
- Perceptrón multicapa con bag of words (tf-idf)
- Red neuronal convolucional
- Red neuronal LSTM

y se utilizaron tres datasets distintos para probar cada uno de ellos:
- Un conjunto de datos de tweets en inglés ya clasificados, extraídos del sitio kaggle (accesible desde [aquí](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis))
- Un conjunto de tweets extraidos mediante *web scraping* (véase el [directorio de scraping](./src/tweetScrapingNodejs) y clasificados por nosotros
- Un conjunto de datos ya etiquetado proveniente de otro paper, [*Balotaje Argentina 2015 a partir de un análisis de sentimiento de tweets* de *Daniel Robins*](https://arxiv.org/ftp/arxiv/papers/1611/1611.02337.pdf) y colaboradores

### Resultados
Para el caso del dataset de tweets del balotaje presidencial argentino del año 2015, se obtuvieron los siguientes resultados:

**Modelo** | **Accuracy** | **Loss**
:---: | :---: | :---: 
**MPL con capa de embedding** | ***0.895*** | ***0.295***
**MLP con bag of words** | *0.894* | *0.301*
**Red neuronal convolucional** | *0.893* | *0.296*
**Red neuronal LSTM** | *0.355* | *1.097*


