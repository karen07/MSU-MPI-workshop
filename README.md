# MSU MPI workshop

This repository contains a 3D wave-equation solver developed as an MSU parallel-programming workshop project.

The computational domain is decomposed between MPI processes, while OpenMP is used inside each process for shared-memory parallelism. Neighboring MPI ranks exchange boundary layers with non-blocking point-to-point communication.

The program can write VTK snapshots for visualization and prints the numerical error and execution time, making the project useful as a compact example of hybrid MPI + OpenMP numerical code.

## Описание

Этот репозиторий содержит решатель трехмерного волнового уравнения, разработанный как проект практикума МГУ по параллельному программированию.

Вычислительная область распределяется между процессами MPI, а OpenMP используется внутри каждого процесса для параллельных вычислений с общей памятью. Соседние процессы MPI обмениваются граничными слоями с помощью неблокирующих двухточечных операций.

Программа умеет записывать снимки VTK для визуализации и выводит численную ошибку и время выполнения, поэтому проект можно использовать как компактный пример гибридного численного кода на MPI + OpenMP.

## Сборка

Нужны CMake, реализация MPI и поддержка OpenMP.

```sh
cmake --preset release
cmake --build --preset release
```

Исполняемый файл:

```text
build/release/mpi-waves
```

## Запуск

Количество MPI-процессов должно быть степенью двойки. Первый необязательный аргумент задает размер сетки по всем трем пространственным направлениям.

```sh
mkdir -p plot
mpirun -np 4 ./build/release/mpi-waves 127
```

Без аргумента используется размер сетки, заданный в исходном коде.

Программа выводит параметры расчета, L2 error и суммарное время работы. VTK-файлы записываются в каталог `plot/` и могут быть открыты, например, в ParaView.

## Реализация

MPI используется для разбиения трехмерной области и обмена граничными слоями между соседними блоками. OpenMP распараллеливает вычислительные циклы внутри каждого MPI-процесса.

Это учебный/экспериментальный solver: параметры численной схемы и формат вывода намеренно находятся непосредственно в коде, чтобы реализацию было проще изучать и менять.
