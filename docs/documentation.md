# Dokumentácia riešenia
**Autori: Marek Drgoňa, Daniel Pekarčík**

Cieľom projektu je návrh a implementácia neurónovej siete, ktorá je schopná odporúčať v doméne vtipov. Naša úloha má charakter 
regresie, pretože naša sieť predikuje aké hodnotenie dá používateľ X vtipu Y. Hodnotenia sú z intervalu reálnych čísiel <-10, 10>

## Úvod
Úvod spoločne s motiváciou a opisom existujúcich prác je uvedený v súbore [Project_proposal.ipynb](https://github.com/mdrgona/nsiete_project/blob/master/Project_proposal.ipynb).

## Analýza dát
Analýza dát je vypracovaná v súbore [Data_analysis.ipynb](https://github.com/mdrgona/nsiete_project/blob/master/notebooks/Data_analysis.ipynb).

## Trénovacia rutina
Trénovanie modelu pozostáva na najvyššej úrovni z nasledujúcich krokov:
1. Načítanie a spracovanie dát
2. Trénovanie modelu
3. Vyhodnotenie

V ďalšej časti si opíšeme jednotlivé kroky trénovacej rutiny detailnejšie.

### Načítanie a spracovanie dát:
Dataset je vo formáte .csv. Jeho charakteristika je uvedená v rámci analýzy dát. Po načítaní rozdelíme dataset 
na trénovaciu a testovaciu podmnožinu, v pomere 80:20. ID používateľov a vtipov nastavíme, 
aby boli z intervalu <0,počet používateľov (resp. vtipov)>. 
Následne neurónovú sieť trénujeme použitím iba trénovacej podmnožiny, ktorá sa ešte ďalej rozdelí na trénovaciu a validačnú 
v pomere 90:10.

Celkové rozdelenie datasetu je teda naslednovné:
* **Trénovanie:  72%** 	(754 974 záznamov)
* **Validácia: 8%**	 (83 886 záznamov)
* **Testovanie: 20%** (209 716 záznamov)

### Trénovanie modelu
Popri trénovaní máme pre všetky varianty modelov nastavené nasledovné parametre rovnako:
* optimizer: Adam
* loss: mean absolute error

Používame viacero modelov (a ich variantov):

#### 1a) Viacvrstvovy perceptron 1 (MLP-1)
Tento model sme navrhli ako prvý a predstavuje východiskový variant (angl. baseline). 
Typ a počet vrstiev, počet neurónov v rámci vrstiev a ďalšie parametre boli skúšané náhodne (a manuálne).
Architektúra NN je na obrázku

**TODO**

#### 1b) Viacvrstvovy perceptron 2 (MLP-2)
Pôvodný variant MLP sme upravili podľa existujúcej práce [1]. Úprava spočívala najmä v pridaní zopár 
nových skrytých vrsiev (dropout a batch-normalization) a zvýšeniu počtu neuronov na vrstách. 
Očakávame zlepšenie pôvodného variantu.
Architektúra NN je na obrázku

**TODO**

#### 2) GMF
GMF (General matrix factorization) predstavuje deep learning prístup ku klasickej faktorizácii matíc. 
Architektúra NN je na obrázku

**TODO**

#### 3) MLP-2 + GMF
Na základe existújucich riešení bolo zistené, že spojenie predošlých 2 typov modelov zlepšuje výsledky. 
Preto sme sa rozhodli aj my navrhnuté modely spojiť do jedného celkového.
Architektúra NN je na obrázku

**TODO**

#### SVD
SVD predstavuje state-of-the-art pristup v doméne odporúčaní, ktorý sme použili s cieľom porovnať naše riešenia.

### Vyhodnotenie
Natrenované modely sme vyhodnocovali použitím testovacej podmnožiny datasetu. Pre každý záznam v testovacich dátach
bolo predikované hodnotenie a následne overené, ako veľmi sa líši od očakavaného hodnotenia.

Modely a ich varianty sme vyhodnocovali pomocou nasledujúcich metrík:
* Mean absolute error (MAE)
* Precision@10

## Výsledky experimentov

### Experiment 1 (MLP-1)

Mean absolute error: 3.3987
Precision@10:    : 0.0603
 

### Experiment 2 (MLP-2)

Mean absolute error: 3.4636
Precision@10:    : 0.0633

### Experiment 3 (GMF)

### Experiment 4 (MLP-2 + GMF)

### Vysledok SVD

Prehľadná tabuľka na konci



## Referencie
[1] [Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web (WWW '17). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, Switzerland, 173-182. DOI: https://doi.org/10.1145/3038912.3052569](https://dl.acm.org/citation.cfm?id=3052569)
