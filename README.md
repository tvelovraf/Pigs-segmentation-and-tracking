ОПИСАНИЕ

Файл для запуска алгоритма - detect.py. 
В качестве двух аргументов используются:
--source путь к тестовому видео;
--output путь для сохранения результата (по умолчанию создаётся папка ./inference/output/ в корне).

Результат сохраняется в виде видео с наложенными боксами, масками, номерами и активностью животных, их количеством для каждого кадра

УСТАНОВКА ПАКЕТОВ

conda create --name название_окружения 

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install python=3.7.9

pip install -r requirements.txt

ПРИМЕР ЗАПУСКА ПРОЕКТА

detect.py --source путь_к_видео_для_обработки/видео.mkv --output путь_к_папке_для_сохранения
