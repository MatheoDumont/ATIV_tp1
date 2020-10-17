# ATIV_tp1

Dépot pour le premier tp d'Analyse d'Image du master ID3D pour le groupe formé par *Yann-Situ Gazull* et *Mathéo Dumont*.
Notre rapport (en anglais) se trouve [ici](TP1_ATIV.pdf).

# Compilation
Pour compiler, vous devrez avoir `cmake` ainsi que `opencv 4.2` installé.

A la racine du projet, faites :
```
mkdir build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release .. && make && ./tp1
```

# Arborescence

Vous pouvez utilisez le `main.cpp` pour simplement visualiser les résultats, il propose tout le nécéssaire pour tester les fonctions implémentées, cités plus bas.

* `kernel.h` contient les fonctions de base : la convolution et le calculs des gradients et angles, bidirectionel et mutli-directionel.
* `seuil.h` contient les fonctions de seuillages demandées : seuille global, locale et hystérésis.
* `contour.h` contient les fonctions de fermeture de contour et d'affinage en utilisant l'érosion, la dilation et une autre méthode dérivée `dilatation_contour()`.
* `path_contour.h` propose une méthode de fermeture de contour récursive qui suit le contour depuis un pixel à l'aide du gradient.
