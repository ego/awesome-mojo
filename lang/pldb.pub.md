# Let's do some research on lang topic!

Recently I found out about *Programming Language DataBase*

[PLDB](https://pldb.pub)

All dataset here

https://pldb.pub/pldb.json


```shell
python3 -m venv .venv
. .venv/bin/activate

pip install ipython pandas

ipython3
```

```python
import pandas as pd

with open("lang/lang.csv") as fd:
    data = []
    for i, line in enumerate(fd.readlines()):
        if i == 0:
            continue
        cols = [x.strip() for x in line.split(" ") if x.strip()]
        if len(cols) > 4:
            cols = [" ".join(cols[:len(cols)-3])] + cols[-3:]
        title, appeared, type_, rank = cols
        data.append(
            {
                "title": title,
                "appeared": int(appeared),
                "type": type_,
                "rank": int(rank),
            }
        )

df = pd.DataFrame(data)

print("New Tech")
new_items = df.sort_values(["appeared", "rank"], ascending=[False, True])
print(new_items[:50].to_string(index=False))
```

New Tech
                  title  appeared             type  rank
                    TQL      2023    queryLanguage  1747
            Scrapscript      2023               pl  2063
                SectorC      2023         compiler  4512
                Speedie      2022               pl   387
               Markwhen      2022       textMarkup   434
                   PRQL      2022    queryLanguage   436
                   Djot      2022       textMarkup   612
                   Jakt      2022               pl   659
                    Zuo      2022               pl   872
           MarkovJunior      2022               pl   895
                 Melody      2022               pl   967
                   Mojo      2022               pl   968
                    Dak      2022               pl  1050
                 HeLang      2022               pl  1108
             Violent ES      2022               pl  1139
                    erg      2022               pl  1155
                  Cyber      2022               pl  1167
                 Pomsky      2022               pl  1374
Uniform eXchange Format      2022     dataNotation  1442
                noulith      2022               pl  1454
                   Goal      2022               pl  1592
                    Lil      2022               pl  1593
                 [x]it!      2022     dataNotation  1606
                   Vely      2022               pl  1620
                 Mangle      2022               pl  1662
                   GLMS      2022               pl  1713
                  Kamby      2022               pl  1836
                   YESS      2022         protocol  1843
                   Wing      2022               pl  1874
                 Pycket      2022               pl  1927
               Astatine      2022               pl  2005
                   JCOF      2022     dataNotation  2056
               Fardlang      2022          esolang  2057
                   Kami      2022       textMarkup  2180
                     fp      2022               pl  2246
                   QOIR      2022 binaryDataFormat  2311
                  Edina      2022               pl  2409
                   cosh      2022               pl  2415
                   Cane      2022  musicalNotation  2487
                  Lesma      2022               pl  2502
                   Mewl      2022          esolang  2509
             Storymatic      2022               pl  2511
                   Fern      2022               pl  2628
               Broccoli      2022          esolang  2722
                 SQHTML      2022               pl  2790
                  Jesth      2022     dataNotation  2797
        Linked Markdown      2022       textMarkup  2904
            EverParse3D      2022              idl  3080
                 NumPad      2022           editor  3153
   Interleaved Notation      2022               pl  3163


```python
print("New Compiler")
compiler = df[df["type"] == "compiler"]
compiler = compiler.sort_values(["appeared", "rank"], ascending=[False, True])
print(compiler[:20].to_string(index=False))
```

New Compiler
             title  appeared     type  rank
           SectorC      2023 compiler  4512
             Kefir      2021 compiler  3333
           chibicc      2019 compiler   853
              Deno      2018 compiler   300
   tinygo-compiler      2018 compiler  3309
 asterius-compiler      2017 compiler  1140
             tarot      2017 compiler  3277
          psyche-c      2016 compiler  1667
          binaryen      2015 compiler   833
             Numba      2012 compiler   482
     jsil-compiler      2010 compiler  1090
   Roslyn compiler      2009 compiler   707
               Xoc      2008 compiler  3416
            Stalin      2006 compiler  2154
            LuaJIT      2005 compiler   902
 polyglot-compiler      2003 compiler  2275
   Tiny C Compiler      2001 compiler   719
      JAL compiler      2000 compiler  1293
               GHC      1992 compiler  2935
python-cl-compiler      1991 compiler  4785

```python
print("New Lang")
df_pl = df[df["type"] == "pl"]
lang = df_pl.sort_values(["appeared", "rank"], ascending=[False, True])
print(lang[:50].to_string(index=False))
```

New Lang
                title  appeared type  rank
          Scrapscript      2023   pl  2063
              Speedie      2022   pl   387
                 Jakt      2022   pl   659
                  Zuo      2022   pl   872
         MarkovJunior      2022   pl   895
               Melody      2022   pl   967
                 Mojo      2022   pl   968
                  Dak      2022   pl  1050
               HeLang      2022   pl  1108
           Violent ES      2022   pl  1139
                  erg      2022   pl  1155
                Cyber      2022   pl  1167
               Pomsky      2022   pl  1374
              noulith      2022   pl  1454
                 Goal      2022   pl  1592
                  Lil      2022   pl  1593
                 Vely      2022   pl  1620
               Mangle      2022   pl  1662
                 GLMS      2022   pl  1713
                Kamby      2022   pl  1836
                 Wing      2022   pl  1874
               Pycket      2022   pl  1927
             Astatine      2022   pl  2005
                   fp      2022   pl  2246
                Edina      2022   pl  2409
                 cosh      2022   pl  2415
                Lesma      2022   pl  2502
           Storymatic      2022   pl  2511
                 Fern      2022   pl  2628
               SQHTML      2022   pl  2790
 Interleaved Notation      2022   pl  3163
               Radish      2022   pl  3301
                 Cish      2022   pl  3524
                Verse      2022   pl  3624
                Argon      2022   pl  3638
             TeaSharp      2022   pl  3649
                Blade      2022   pl  3861
                Bolin      2022   pl  3862
               Cosmos      2022   pl  3869
               Gaiman      2022   pl  3887
             LinkText      2022   pl  3897
              Peridot      2022   pl  3912
Simple Stackless Lisp      2022   pl  3925
           SuperForth      2022   pl  3930
                 MiKe      2022   pl  4334
               Qunity      2022   pl  4480
                 Jule      2021   pl   615
               Triton      2021   pl   654
                Slope      2021   pl   701
              Swallow      2021   pl  1200
