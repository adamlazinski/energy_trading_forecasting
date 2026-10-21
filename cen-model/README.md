# cenmodel — replikacja i prognoza ceny niezbilansowania (CEN)

Samodzielny pakiet: pobiera dane z API PSE, odtwarza CEN regułą WDB i prognozuje
**przedziały** CEN na `horizon_min` (domyślnie 60) minut przed okresem dostawy.
Możesz podpiąć własne dane intraday. Pakiet sam pilnuje, żeby model widział tylko to,
co było znane w chwili bramki.

Podstawy: `../docs/CEN_replikacja.md` (pola i algorytm), `../RESEARCH_LOG.md` F37–F38.

---

## 1. Instalacja (raz)

Potrzebny Python 3.10–3.14.

```bash
git clone https://github.com/adamlazinski/energy_trading_forecasting.git
cd energy_trading_forecasting/cen-model

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # ~2 min, wszystko ma być "passed"
```

Na macOS LightGBM wymaga `libomp`. Jeśli import `lightgbm` zgłasza błąd, uruchom
`brew install libomp`.

Wszystkie komendy uruchamiaj **z katalogu `cen-model`** i z aktywnym `.venv`.

## 2. Pierwsze uruchomienie, bez Twoich danych (~10 min)

```bash
python -m cenmodel init        # tworzy config.yaml i katalog my_data/
python -m cenmodel fetch       # pobiera dane PSE od 2024-06-14 (~3-5 min)
python -m cenmodel verify      # replikacja CEN; L1 i L3 mają dać 100.0%
python -m cenmodel doctor      # podsumowanie: co jest, czego brakuje
```

Opcjonalnie pełna replikacja CKOEB z drabinki ofert za jeden miesiąc (~10 s na dzień):

```bash
python -m cenmodel fetch --ladder 2026-09
python -m cenmodel verify --month 2026-09
```

## 3. Twoje dane

Wrzuć pliki do `my_data/` i opisz je w `config.yaml` w sekcji `own_data`.
Przykłady formatów są w `examples/`. Obsługiwane są dwa rodzaje plików, CSV albo parquet.

### `kind: realtime` — pomiary (niezbilansowanie, aktywacje aFRR, ACE, częstotliwość...)

```
ts,imbalance_mw,afrr_up_mw
2026-10-18 13:00:00+02:00,-412.5,180.0
2026-10-18 13:01:00+02:00,-398.1,175.2
```

- `time_col` = **chwila, w której wartość była dla Ciebie znana**. To najważniejsza
  rzecz w całym pakiecie. Jeśli znacznik oznacza początek okresu uśredniania,
  ustaw `known_delay_min` na długość tego okresu (plus opóźnienie dostawy danych).
  Przy wątpliwościach daj większy zapas: przedziały wyjdą trochę szersze, ale uczciwe.
- Dowolna rozdzielczość (sekundy, minuty, kwadranse), dowolna liczba kolumn liczbowych.
- Na kolumnę model liczy trzy cechy: ostatnią wartość przed bramką, zmianę w godzinie
  przed bramką i średnią z ostatnich 60 min. Wartość starsza niż `max_staleness_min`
  liczy się jako brak danych.

### `kind: forecast` — prognozy (wiatr, PV, zapotrzebowanie, ceny...)

```
issued_at,target_ts,wind_mw,pv_mw
2026-10-18 11:00:00Z,2026-10-18 13:00:00Z,5120,3400
```

- `issued_col` = kiedy prognozę wydano; `target_col` = **początek** okresu, którego dotyczy.
- Prognozy godzinowe: `target_resolution_min: 60`. Każdy wiersz trafi do 4 kwadransów.
- Dla okresu T model bierze najnowszą prognozę wydaną najpóźniej w chwili bramki.

### Czas i format

- Daty ze strefą (`+02:00`, `Z`) są czytane automatycznie. Daty bez strefy wymagają
  `timezone: Europe/Warsaw` albo `timezone: UTC`. Najbezpieczniej podawać UTC.
- Pliki CSV z polskiego Excela: `sep: ";"` i `decimal: ","`. Daty w formacie
  `18.10.2026 13:45` są czytane jako dzień.miesiąc.
- Excela (`.xlsx`) nie wczytuję. Zapisz go jako CSV.

Sprawdzenie danych:

```bash
python -m cenmodel check-data
```

Komenda pokazuje zakres, przerwy i pokrycie każdej cechy oraz jej korelację z SK na bramce.
Korelacja **bliska 1.0** prawie zawsze oznacza, że znacznik czasu jest za wczesny
i dane „widzą przyszłość” (wyciek). Popraw `time_col` / `known_delay_min`.

## 4. Czy Twoje dane pomagają? Backtest

```bash
python -m cenmodel backtest                       # ostatnie 60 dni jako test
python -m cenmodel backtest --test-start 2026-08-01 --simulate-rt-sk
```

Backtest trenuje model na danych sprzed testu i porównuje zestawy cech:

| zestaw | co zawiera |
|---|---|
| `publiczne` | kalendarz, CSDAC, opublikowane SK/CEN (D+1) |
| `publiczne+Twoje` | + cechy z `own_data` |
| `...+idealne SK` | (`--simulate-rt-sk`) + prawdziwe SK zakończonych kwadransów; górna granica tego, co może dać sygnał niezbilansowania |

Trzy modele: `structural` (scenariusze SK przez regułę WDB), `direct` (GBM kwantylowy
wprost na CEN) i `blend` (średnia obu). Kolumny raportu:

- **pinball**: średnia strata kwantylowa [zł/MWh]; mniej znaczy lepiej. To główna miara.
- **pokr.P10-90**: odsetek okresów, w których CEN wpadł między q10 a q90. Powinno być ~0.80.
- **pokr.P5-95**: powinno być ~0.90.
- **pinball surowy**: wynik przed kalibracją. Pokazuje, ile wnosi kalibracja konformalna.

Raport trafia do `work/reports/backtest.txt`, a predykcje do `work/reports/backtest_preds.parquet`.
Model wybierasz w `config.yaml` (`model.final`) na podstawie backtestu.

## 5. Używanie na żywo

```bash
python -m cenmodel train                    # raz w tygodniu lub po zmianie own_data (~3-5 min)
python -m cenmodel predict --refresh        # okresy, których bramka minęła, a dostawa jeszcze nie
python -m cenmodel predict --start "2026-10-20 08:00" --end "2026-10-20 12:00"
python -m cenmodel predict --refresh --scenarios   # + 200 scenariuszy CEN na okres (np. do BESS)
```

Wynik zapisuje się do CSV w `work/predictions/` (albo do pliku z `--out`):

| kolumna | znaczenie |
|---|---|
| `period_start_utc`, `period_start_warsaw` | początek okresu dostawy |
| `gate_warsaw` | chwila, na którą zbudowano cechy |
| `q05` … `q95` | kwantyle CEN [zł/MWh]; `q50` = mediana, `q10`–`q90` = przedział 80% |
| `p_short` | prawdopodobieństwo, że system jest krótki (SK<0, cena ciągnięta w górę) |
| `sk_q50_mwh` | mediana prognozy SK |
| `csdac`, `cen_actual` | cena day-ahead; faktyczne CEN, jeśli już opublikowane |

Dlaczego `train` nie używa ostatnich 16 dni: na nich `predict` kalibruje przedziały,
porównując prognozy modelu z faktycznym CEN. Działa to tylko na danych, których model
nie widział, więc kalibracja jest uczciwa. Gdy kalibracji brakuje, `predict` wyświetla
ostrzeżenie. Wtedy przedziały będą za wąskie.

`predict` ostrzega też, gdy Twoje dane są nieaktualne na bramce. Model działa wtedy
dalej, ale bez Twojego sygnału.

## 6. Gdy coś nie działa

1. `python -m cenmodel doctor` pokazuje stan instalacji, API, cache, danych i modelu.
2. Komunikat zaczynający się od `BŁĄD:` mówi, co poprawić (konfiguracja, plik, sieć).
3. „Nieoczekiwany błąd” to bug w pakiecie. Skopiuj cały tekst i wynik `doctor`.
   Opcja `--debug` pokazuje pełny traceback.
4. `fetch` przerwany w połowie można uruchomić ponownie. Pobrane miesiące zostają w cache.
5. Po zmianie `own_data` lub `horizon_min` trzeba ponownie uruchomić `train`.
   `predict` sam to wykrywa.
6. Nowy rok: dopisz CORMin do `cor_min` w `config.yaml`, gdy PSE go ogłosi.
   Do tego czasu używana jest ostatnia znana wartość.

## 7. Jak działa model (skrót)

```
SK_s     ~ GBM kwantylowy na SK (21 kwantyli) -> 200 scenariuszy na okres
V_s      = a·(−SK_s) + b                    wolumen aktywacji w kierunku pomagającym
share_s, dprem_s, eps_s  ~ analogi z historii o tym samym znaku, bloku godzin i |SK|:
                            udział aFRR, premia aFRR (CEBśr − CEB_PP), błąd krzywej
CKOEB_s  = g(ZEB_PP_s, x) + eps_s           krzywa podaży, monotoniczna w ZEB_PP
CEN_s    = reguła WDB 13.1(SK_s, CKOEB_s + CORMin + dprem_s, CSDAC)
kwantyle ze scenariuszy -> przesunięcie konformalne z ostatnich 14 dni poza próbą
```

Wyniki z badań (test 08–10.2026, H=60): przy samych danych publicznych pinball wynosi
~129. Z idealnym sygnałem niezbilansowania w czasie rzeczywistym spada do ~118
(`--simulate-rt-sk`). Bez kalibracji konformalnej pokrycie P10–90 to tylko 0.42–0.55;
z kalibracją ~0.80. Szczegóły: F38 w `../RESEARCH_LOG.md`.

Struktura kodu:

| plik | rola |
|---|---|
| `cenmodel/pse.py` | klient API PSE, cache miesięczny, drabinka i oferty dziennie |
| `cenmodel/wdb.py` | reguła CEN, COR z LOLP, CKOEB z drabinki, weryfikacja |
| `cenmodel/panel.py` | panel 15-min, cechy publiczne bez wycieku |
| `cenmodel/owndata.py` | wczytanie i walidacja Twoich danych, cechy na bramce |
| `cenmodel/model.py` | model strukturalny, GBM, kalibracja, metryki |
| `cenmodel/ladder.py` | krzywe cena–wolumen z drabinek, PCA, prognoza krzywej |
| `cenmodel/cli.py` | komendy |
| `tests/` | testy reguł, braku wycieku i przebiegu end-to-end na danych syntetycznych |

## 8. Drabinki ofert: krzywe cena–wolumen (osobny moduł)

Do badania i modelowania poziomów cen ofertowych. Każdy kwadrans i kierunek
zamieniamy na **cenę przy stałym skumulowanym wolumenie** (25, 50, 100, …, 4000 MW,
w porządku merit-order: G od najtańszej, D od najdroższej). Tak samo PSE wyznacza
CKOEB (przecięcie na |ZEB_PP|×4 MW), więc pipeline sam sprawdza, czy krzywe są dobre.

```bash
python -m cenmodel fetch                              # dane rozliczeniowe (jeśli jeszcze nie)
python -m cenmodel fetch --ladder 2025-11..2026-09    # drabinka BPKD/BO, ~10 s/dzień
python -m cenmodel fetch --offers 2025-11..2026-09    # opcjonalnie: wszystkie oferty poeb-rbn
python -m cenmodel ladder --test-days 60              # -> reports/ladder.txt
```

Dwa źródła:

| | `oeb-bpkdbo` (`--ladder`) | `poeb-rbn` (`--offers`) |
|---|---|---|
| co | pasma w kierunku aktywacji, cena CDO (z korektą rozruchową) | wszystkie oferty: moc + cena G i cena D |
| publikacja | D+1 | ~2 min po okresie |
| rola | **cel**: krzywa, która wyznaczyła CKOEB | cecha dostępna na bramce |

Raport `reports/ladder.txt`:

1. **kontrola** — CKOEB z drabinki vs opublikowany (powinno być ~100%; inaczej
   coś jest nie tak z danymi).
2. **opis krzywych** — dla każdego punktu siatki: P10/mediana/P90, jak często drabinka
   się kończy, autokorelacja 15 min / 24 h / 7 dni, korelacja z CSDAC; profil
   godzinowy i miesięczny; udział korekty rozruchowej na paśmie krańcowym.
3. **PCA** krzywej (asinh ceny) — zwykle 3 czynniki: poziom, nachylenie, ogon.
4. **prognoza krzywej** na ostatnie `--test-days` dni: D-2 ta sama pora, mediana
   7 dni, ostatnia krzywa rbn sprzed bramki, regresja (kalendarz + CSDAC + lagi).
   Ocena: prognozowaną krzywą przecinamy na **rzeczywistym** wolumenie i porównujemy
   z CKOEB — to mierzy błąd samej krzywej, bez błędu prognozy wolumenu.
   „Wyrocznia” (prawdziwa krzywa) to błąd samej siatki, dolna granica.

Pliki: `reports/ladder_curves.parquet` (ts, dir, p25…p4000, ladder_mw, p_top,
ckoeb_hat, need_mw, startup_marginal), `ladder_curves_rbn.parquet`,
`ladder_eval.parquet` (prognozy per okres i model). Na tym budujesz własne modele.

Bez wycieku: drabinka BPKD/BO dnia D jest jawna dopiero D+1, więc jako wejście
prognozy używana jest z opóźnieniem ≥ 2 doby; rbn z opóźnieniem
⌈(H+20)/15⌉ kwadransów (H = `horizon_min`).

Wyniki na 11.2025–07.2026 (test 05–07.2026, kierunek G, MAE przy rzeczywistym
wolumenie): mediana 7 dni **75 zł**, regresja 78, D-2 90, rbn wprost 201, wyrocznia 6.
Krzywa rbn ma inny poziom niż BPKD/BO (najpewniej zawiera pasma, których przy bieżącym grafiku
nie da się aktywować), więc wprost się nie nadaje — co najwyżej jako cecha.
