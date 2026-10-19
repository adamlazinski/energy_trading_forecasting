# Replikacja CEN od zera z API PSE

Podstawa prawna: WDB (tekst jednolity z Zmianami 3/2025), pkt 13.2–13.3.
Implementacja referencyjna: `pl-cen-forecaster/src/cen_reconstruct.py`.
Zweryfikowane na 82 176 okresach (2024-06-14 → 2026-10-17), w tym na
pełnym wrześniu 2026 (2 880 okresów).

## 0. Co da się, a czego nie da się odtworzyć

| Krok | Wynik (IX 2026) | Uwagi |
|---|---|---|
| CKOEB z drabinki BPKD/BO | 100,00% co do grosza | |
| COR z LOLP(RO), CORDMax, CORMin | 100,00% (cała historia) | |
| CEB_PP = CKOEB + COR | 100,00% | |
| CEN z CEBśr, SK, CSDAC | 100,00% | |
| CEBśr bez aFRR w kierunku pomagającym | 99,8% (1 okres CUA) | |
| CEBśr z aFRR | ~2% dokładnie, MAE ~37 PLN | **niewykonalne z danych publicznych**, patrz pkt 6 |

We wrześniu 2026 aFRR występuje w 80% okresów, więc pełne CEN co do
grosza da się odtworzyć tylko w ~20% okresów. W pozostałych daje się
dokładnie wszystko oprócz członu aFRR.

## 1. Pobieranie danych

Baza: `https://api.raporty.pse.pl/api/<slug>`, bez autoryzacji.
Filtr: `?$filter=business_date ge '2026-09-01' and business_date le '2026-09-30'&$first=50000`.
Spacje kodujemy jako `%20`, nie `+`, bo inaczej API zwraca 400. Trzeba
obsłużyć paginację przez `nextLink`. Drabinkę (`oeb-bpkdbo`) najlepiej
pobierać dzień po dniu, bo ma ok. 13–400 tys. wierszy na dobę.

| Slug | Pola | Znaczenie (symbol WDB) | Jednostka |
|---|---|---|---|
| `crb-rozl` | `cen_cost` | CEN (cel do weryfikacji) | zł/MWh |
| | `ckoeb_cost` | CKOEB | zł/MWh |
| | `ceb_pp_cost` | CEB_PP | zł/MWh |
| | `ceb_sr_cost` | CEBśr = średnia ważona CEB (CEB̄) | zł/MWh |
| | `ceb_sr_afrrg_cost`, `ceb_sr_afrrd_cost` | cena aFRR G / D (agregat 15 min) | zł/MWh |
| `price-cost` | `cor_pp_cost` | COR (**osobno dla każdego OREB, nie stała dzienna**) | zł/MWh |
| | `contracting` | short/long (kontrola znaku SK) | – |
| `zeb-rozl` | `zebpp` | ZEB_PP (znak = kierunek) | **MWh** |
| | `pzeb_afrrg_cost`, `pzeb_afrrd_cost` | PZEB_aFRR G (≥0) / D (≤0); **to wolumeny, mimo sufiksu `_cost`** | **MWh** |
| `sk` | `sk_cost` | SK (stan zakontraktowania; <0 = short) | MWh |
| `csdac-pln` | `csdac_pln` | CSDAC | zł/MWh |
| `oeb-bpkdbo` | `cdo`, `ofp`, `activ_direction` | pasma planu BPKD/BO: cena CDO, moc pasma, kierunek G/D | zł/MWh, **MW** |

Klucz łączenia to `dtime_utc` (lub `dtime`). Uwaga: `dtime` oznacza
**koniec** okresu (`dtime 00:15` ↔ `period 00:00 - 00:15`).

Pułapki:
- W drabince używamy **`cdo`, nie `ofc`**. `ofc` to cena ofertowa,
  a `cdo` to cena skorygowana (np. o koszty uruchomienia); różnią się
  w ~4% wierszy.
- Kolumna `sort` **nie** jest porządkiem merit-order, nie sortujemy po niej.
- `activ_direction` jest jeden na okres i zawsze zgadza się ze znakiem
  `zebpp` (G ⇔ ZEB_PP>0).

## 2. Krok A — CKOEB z drabinki (WDB 13.2(3)–(5))

Dla każdego okresu t:

1. Weź wszystkie wiersze `oeb-bpkdbo` dla t (bez deduplikacji, patrz niżej).
2. Posortuj po `cdo`: **rosnąco, gdy G**, **malejąco, gdy D**.
3. Policz `cum = cumsum(|ofp|)`.
4. `need = |zebpp| × 4` (MWh na 15 min → MW).
5. **CKOEB = `cdo` pierwszego pasma, dla którego `cum ≥ need`.**
6. Jeśli `zebpp = 0`, to CKOEB = CUA. Tej ceny nie da się odtworzyć,
   ale przypadek w praktyce nie występuje.

**Naprawa replikacji w API.** W ok. 46 dniach (m.in. 2024-09, 2025-04,
2025-11/12) API zwraca drabinkę każdego okresu powieloną k razy
(k ≈ 9–14); ponowne pobranie daje to samo. Zwykły `drop_duplicates` jest
błędny, bo identyczne pasma różnych jednostek są legalne. Poprawnie:

```
cnt   = liczność każdej unikalnej krotki (cdo, ofp, activ_direction) w okresie
dup   = 1 − (liczba unikalnych krotek / liczba wierszy)
jeśli dup > 0.3:  k = NWD(cnt);  cnt = cnt / k
```

We wrześniu 2026 problem nie wystąpił.

## 3. Krok B — COR i CEB_PP (WDB 12.3–12.5, 13.2(2))

COR można wziąć gotowy z `price-cost.cor_pp_cost` albo policzyć od zera.
Wersja liczona daje **100,00% na całej historii**. Dodatkowe dane:

| Slug | Pola | Znaczenie |
|---|---|---|
| `lolp` | `b0..b9` [MW], `p0..p9`, `ojnz_id` | krzywa LOLP dla okresu (b0 = ROMin, już wliczone) |
| `ro-rozl` | `ro_cost` | RO = rezerwa operacyjna wszystkich JG, **MW** (mimo sufiksu `_cost`) |
| `cordmax` | `cordmax_cost` | CORDMax, dzienny, klucz `business_date` |

```
LOLP = 1                                 jeśli RO < b0
     = 0                                 jeśli RO ≥ b9
     = interpolacja liniowa p_i→p_{i+1}  dla b_i ≤ RO < b_{i+1}

COR  = 0                                              jeśli LOLP = 0   (~41% okresów)
     = min(GLC − CKOEB; max(CORMin; CORDMax · LOLP))  w p.p.;  zaokrąglić do 0,01

CEB_PP = CKOEB + COR
```

- CORMin = 10 zł/MW-h indeksowane CPI od 2025 (WDB 12.4(3)): **10,00
  (2024), 10,50 (2025), 10,82 (2026)**. PSE publikuje go na pse.pl,
  a nie w API; powyższe wartości odczytałem z danych.
- GLC to zharmonizowany górny limit ceny energii bilansującej (metodyka
  ACER, przeliczony kursem NBP z d-2). W całej próbie ani razu nie wiąże,
  więc można przyjąć +∞.

## 4. Krok C — CEBśr (WDB 13.3(5), wzory 13.3–13.7)

Liczymy tylko z energii aktywowanej w kierunku, który **pomaga** systemowi:

```
jeśli SK < 0 (short):  Z = max(0, zebpp);  P = max(0, pzeb_afrrg);  C = ceb_sr_afrrg
jeśli SK > 0 (long):   Z = min(0, zebpp);  P = min(0, pzeb_afrrd);  C = ceb_sr_afrrd

jeśli Z + P ≠ 0:   CEBśr = (Z·CEB_PP + P·C) / (Z + P)       ← człon P·C tylko przybliżony
w przeciwnym razie (lub SK = 0):  CEBśr = CUA + COR       ← CUA niepublikowane
```

Wynik zaokrąglamy do 0,01. Gdy P = 0 (brak aFRR w kierunku
pomagającym), CEBśr = CEB_PP **dokładnie**.

## 5. Krok D — CEN (WDB 13.3(1), wzór 13.1)

```
SK > 0:  CEN = min(CEBśr, CSDAC)     (gdy CSDAC = 0: min(CEBśr, −0,01))
SK < 0:  CEN = max(CEBśr, CSDAC)     (gdy CSDAC = 0: max(CEBśr, +0,01))
SK = 0:  CEN = CEBśr
```

Ten krok z opublikowanym `ceb_sr_cost` daje 100,00% na całej historii.
CSDAC jest wiążące w ~10–16% okresów.

## 6. Dlaczego człon aFRR jest niereplikowalny

WDB waży cenę aFRR **per cykl optymalizacji OAaFRR (4 s)**:
Σ_τ PZEB_τ · CEB_τ. PSE publikuje tylko agregaty 15-minutowe:
Σ_τ PZEB_τ (`pzeb_afrr*`) i jedną cenę (`ceb_sr_afrr*`), która **nie**
jest odpowiadającą jej średnią ważoną wolumenem. Dowody (IX 2026):

- W 395 okresach jedynym składnikiem w kierunku pomagającym jest aFRR
  (Z = 0). Wtedy reguła daje CEBśr = średnia ważona z cykli. Opublikowane
  CEBśr równa się `ceb_sr_afrr*` tylko w **0,8%** z nich.
- W ~20% okresów z aFRR CEBśr leży **poza** przedziałem
  [CEB_PP, ceb_sr_afrr*]. Żadna średnia ważona dwóch liczb nie może
  dać takiego wyniku.

W API PSE nie ma danych per cykl. Do dokładnej replikacji potrzebne
byłyby per cykl: polskie PZEB_τ oraz CBMP_τ z PICASSO. Ceny CBMP są
publikowane (ENTSO-E Transparency / PICASSO), ale polskie wolumeny
per cykl publicznie nie są. Najlepsze, co da się zrobić, to wzór z pkt 4
jako przybliżenie (IX 2026: MAE ~30 PLN na CEN, ~30% okresów w 1 PLN).

## 6a. CUA — czego jeszcze nie da się odtworzyć (i dlaczego to bez znaczenia)

CUA wchodzi do ceny tylko wtedy, gdy w kierunku pomagającym nic nie
aktywowano: ani ZEB_PP, ani aFRR. CEBśr = CUA + COR, gdzie CUA to
najniższa CDO dostępnych pasm w górę (SK<0) albo najwyższa CDO pasm
w dół (SK>0). Drabinka `oeb-bpkdbo` zawiera jednak tylko pasma w kierunku
`activ_direction`, czyli przeciwnym do potrzebnego, a kryteria dostępności
pasm (sieć, uwarunkowania JG) nie są publiczne. **Przez 2,3 roku
zdarzyło się to w 7 okresach**, w 2 z nich i tak wiązało CSDAC.
Praktycznie pomijalne.

## 7. Kontrola poprawności (checklista)

1. `sign(sk_cost)` zgadza się z `price-cost.contracting` w 100%.
2. `ckoeb_cost + cor_pp_cost == ceb_pp_cost` w 100%.
3. `rule(sk, ceb_sr_cost, csdac)` zwraca `cen_cost` w 100%. Jeśli nie,
   błąd jest w łączeniu po czasie (początek vs koniec okresu).
4. `activ_direction` zgadza się ze znakiem `zebpp` w 100%.
5. CKOEB z drabinki w 100%. Jeśli wychodzi ~5%, przecinasz drabinkę na
   złym wolumenie: `eb-rozl` zamiast `zebpp`, albo zapominasz o ×4
   (MWh→MW), albo sortujesz po `sort`/`ofc` zamiast po `cdo`.
