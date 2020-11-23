# TPs d'Analyse Traitement Vision 3D

Dépot pour les tps d'Analyse d'Image du master ID3D pour le groupe formé par *Yann-Situ Gazull* et *Mathéo Dumont*.

Rapports (en anglais) :
  * [TP1](rapport_tp1.pdf)
  * [TP2](rapport_tp2.pdf)

# Compilation
Pour compiler, vous devrez avoir `cmake` ainsi que `opencv 4.2` installé.

A la racine du projet, faites :
```
mkdir build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```
# Les exécutables
Sont proposés à exécuter, une fois le projet compilé (dans le répertoire `build/`) :
* `TP1`
* `TP2`
  
# Arborescence

Vous pouvez utilisez les fichiers `mains/` qui sont les main pour chaque TP, par exemple `mains/TP1.cpp` est le main du TP 1, pour simplement visualiser les résultats, il propose tout le nécéssaire pour tester les fonctions implémentées, cités plus bas.

## TP 1

* `kernel.h` contient les fonctions de base : la convolution et le calculs des gradients et angles, bidirectionel et mutli-directionel.
* `seuil.h` contient les fonctions de seuillages demandées : seuille global, locale et hystérésis.
* `contour.h` contient les fonctions de fermeture de contour et d'affinage en utilisant l'érosion, la dilation et une autre méthode dérivée `dilatation_contour()`.
* `path_contour.h` propose une méthode de fermeture de contour récursive qui suit le contour depuis un pixel à l'aide du gradient.

## TP2

* `hough_lines.h` contient les fonctions concernant la transformee de hough pour les lignes
* `hough_cercle.h` contient les fonctions concernant la transformee de hough pour les cercles
