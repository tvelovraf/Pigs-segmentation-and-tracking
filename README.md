# Intro

Решение задачи сегментации, трекинга и оценки активности поросят командой **svinkotrack** в рамках хакатона [AgroHack Code](https://agro-code.ru/hack/task/digital-farm/).  

**Пробный запуск** можно осуществить на [**Google Colab**](https://colab.research.google.com/drive/1-D7cfnjwPrFF92_6qwfvWCrWFpW7JSuT?usp=sharing#scrollTo=AQEXOi6-twGA)

## Installation guide

Предполагается наличие установленной Anaconda. 

Клонируем репозиторий:
```bash
git clone https://github.com/tvelovraf/Pigs-segmentation-and-tracking.git
cd Pigs-segmentation-and-tracking
```

### Установка пакетов

Создаём и активируем новое окружение:

```bash
conda create --name <envname>  
conda activate <envname> 
```
Устанавливаем `Python` и необходимые библиотеки:
```bash
conda install python=3.7.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
``` 
Устанавливаем зависимости из `requirements.txt`:
```bash
pip install -r requirements.txt
```

Теперь проект готов к запуску.

### Запуск
Файл для запуска проекта — `detect.py`.  
В качестве двух аргументов используются:  
* `--source <path/to/test_video>` путь к тестовому видео;  
* `--output <path/to/inference/>` путь к директории для сохранения результата (по умолчанию создаётся папка `./inference/output/` в корне).

Запускаем проект для тестового видео `test_cut.mp4`:
```bash
python detect.py --source test_cut.mp4 --output ./inference/output
```

Результат сохраняется в виде видео с наложенными боксами, масками, номерами, активностью животных и их количеством для текущего кадра. Также сохраняется картинка `activity.png` с графиком активностей свиней за всё время.

## Пример результата работы
<p align="center">
  <img src="./imgs/Movie_1.gif" width="600" />
  <img src="./imgs/Movie_2.gif" width="600" /> 
</p>

