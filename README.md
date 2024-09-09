# netflix-recommender
Рекомендательная система на основе датасета от Netflix

# Описание
Модель использует подход content-based filtering, основываясь на информации о фильме, чтобы предложить похожие.  
Метод TF-IDF (Term Frequency-Inverse Document Frequency) применяется, чтобы преобразовать текстовые данные в числовые векторы.  
После того как все фильмы представлены в виде числовых векторов, мы вычисляем косинусное сходство между этими векторами. Косинусное сходство — это метрика, которая измеряет степень схожести между двумя векторами, основываясь на угле между ними. В нашей модели это позволяет нам определить, насколько похожи фильмы друг на друга. Например, если два фильма имеют высокое косинусное сходство, это означает, что их описания и характеристики схожи.  
Когда пользователь запрашивает рекомендации для определенного фильма, система находит вектор этого фильма и сравнивает его с векторами всех остальных фильмов, используя ранее рассчитанную матрицу косинусного сходства. На основе этого сходства система сортирует все фильмы по степени их похожести на указанный фильм и выводит список наиболее похожих фильмов и ТВ-шоу.  
Более подробно с моделью можно ознакомится в файле NetflixRecSys.ipynb  

# Установка и запуск
1. Скачайте репозиторий  
2. Установите необходимые пакеты  
   ```pip install -r requirements.txt ``` 
4. Запустите приложение  
   ```python app.py  ```
5. В браузере введите  
   ```http://127.0.0.1:5000  ```
6. Введите название фильма или сериала, чтобы увидеть рекомендации похожих тайтлов
![image](https://github.com/user-attachments/assets/ef236851-a282-4ce0-a1f4-99599fc0ba3a)


