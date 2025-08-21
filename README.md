# Konvolúciós neurális háló és OOD detektálás a FashionMNIST adathalmazon

# Projekt áttekintés

Ebben a projektben egy konvolúciós neurális háló (CNN) került betanításra a FashionMNIST adathalmazon, majd a modell teljesítményét kiterjesztve OOD (out-of-distribution) detektálás is vizsgálatra került a MNIST kézírt számok bevonásával.
A cél az volt, hogy a modell ne csak az ismert (ID) osztályokat ismerje fel, hanem bizonytalanság esetén képes legyen jelezni, ha egy minta nem tartozik a tanult eloszláshoz.

# Adatok

FashionMNIST (in-distribution)

Train halmazból véletlenszerűen kiválasztott ~6000 minta (10%)

Teszt halmazból véletlenszerűen kiválasztott ~1000 minta

MNIST (out-of-distribution)

Kézírt számjegyek az OOD vizsgálathoz

# Modell

Konvolúciós neurális háló architektúra:

2 konvolúciós réteg (32 és 64 szűrő, ReLU aktivációval)

Max pooling és Dropout rétegek

2 teljesen kapcsolt réteg (dense)

Tanítás:

Optimizer: Adam

Epoch: 5

Eredmény: ~90% train accuracy, ~86% test accuracy

# OOD detektálás

Wrapper osztály létrehozása, amely ID/OOD címkéket ad a mintákhoz

Az ID minták helyesnek számítanak, ha a predikció egyezik a címkével

Az OOD minták mindig hibásnak számítanak (baseline szabály)

Így a teljes OOD-aware pontosság kb. 43% lett

# Softmax és logit vizsgálatok

A softmax hajlamos „túl magabiztos” predikciókat adni OOD mintákra is

A logit értékek (softmax előtti kimenetek) jobban tükrözik a bizonytalanságot

Küszöbértékek (Thr, Thr_G, Thr_B) meghatározása a logit maximum alapján

10%-os biztonsági korrekció a küszöböknél

# Eredmények

A küszöbértékek alapján ID/OOD detektálás történt

Az eredmények azonban gyengeek voltak (~0.50 körüli pontosság)

Következtetés: a pusztán logit-alapú küszöbölés nem elegendő megbízható OOD detektáláshoz

# Használt technológiák

Python 3

Könyvtárak: torch, torchvision, numpy, matplotlib

# Következtetések

A CNN jól teljesít a FashionMNIST osztályozásban (~86% pontosság).

Az OOD minták detektálása jóval nehezebb feladat.

A softmax kimenet túlzottan magabiztos OOD esetben is.

A logit-küszöb módszer önmagában nem hoz jó eredményt.

További fejlesztési irány lehet: entropia-alapú bizonytalanság mérés, ODIN-módszer, vagy más mély tanulási OOD technikák alkalmazása.
