###

タンパク質は生命の基本要素だ。筋肉、血液、ホルモン、髪の毛など、ヒトの乾燥重量の実に 75％がタンパク質でできている。タンパク質は体内にあらゆる形態で存在し、さまざまに重要な役割を担っている。たとえば骨同士をつなぎ合わせる靱帯や病原体を排除する抗体などだ。生物学に精通するための大きな第一歩は、タンパク質を理解することである。
だが、単に DNA 配列を理解しても、タンパク質の働きは十分にはわからない。むしろタンパク質がどのように折り畳まれているか（フォールディング）を理解する必要がある。鎖状だったタンパク質が折り畳まれて特定の構造を形成することにより、タンパク質は固有の機能を発揮する。
たとえば腱に含まれるコラーゲンは繊維のような構造を持つ。補因子（非タンパク質性の分子）と結合して機能する酵素には、補因子を入れるためのポケットが備わっている。だが、タンパク質が折り畳まれる仕組みは不明なので、たったひとつのタンパク質について、アミノ酸のつながり方から考えられる可能性をしらみつぶしにコンピュータで計算しようとすれば、既知の宇宙の寿命より長い時間がかかるかもしれない。タンパク質がどう折り畳まれるかを解明するのは実に困難で、創薬からプラスチック分解酵素の開発に至るまで、あらゆる進歩の妨げになってきた。
数十年にわたり、科学者たちはタンパク質の折り畳みの仕組みを知るよい方法がないかとずっと考えてきた。そして 1993 年、タンパク質の折り畳み問題を解決するため、タンパク質立体構造予測コンテスト（CASP）を隔年開催することにした。タンパク質の立体構造を最も正確に予測した者が勝者だ。この分野の科学者たちは激しく競い合っているが、互いに緊密な関係にあり、CASP はすぐに研究者たちのベンチマークとなった。進歩は着実だったが、タンパク質の折り畳み問題が解決しそうな兆しはなかった。
2018 年の CASP の 13 回目大会「CASP13」は、ヤシの木が立ち並ぶメキシコのリゾート地、カンクンで開かれた。定評ある 98 チームを下して優勝したのは、門外漢で実績ゼロのディープマインドのチームだった。
優勝したディープマインドが使った AI、AlphaFold（アルファフォールド）は、2016 年に行われた 1 週間のハッカソンで、私のグループから始まったプロジェクトだ。計算生物学において画期的な結果を出すまでに進化した AlphaFold は、AI と生命工学がともに高速で進歩している最高の例だ。
第 2 位は高く評価されているチャン・グループ（Zhang group）だったが、このチャン・グループは最も予測が難しいとされた 43 個のタンパク質ターゲットのうち 3 つを予測した。だが、AlphaFold は 25 個を予測した。ほかの参加チームよりもはるかに速く、わずか数時間でこれを成し遂げたのだ。超優秀な専門家が参加する権威ある大会で実績ゼロのチームが優勝したことに、誰もがびっくりした。著名なシステム生物学者モハメド・アルクライシは、「何が起こったんだ?」と疑問を示した。
AlphaFold は深層生成ニューラルネットワークを使い、既知のタンパク質を学習させることで、DNA 配列からタンパク質の立体構造を予測できるようにした。AlphaFold の新しいモデルではアミノ酸残基の位置関係をさらに正確に推測できるようになった。
タンパク質の折り畳み問題を解決するのに必要だったのは、液体窒素を用いるクライオ電子顕微鏡法など以前からある専門的な手法ではなかったし、製薬や従来のアルゴリズム法の専門知識でもなかった。必要だったのは、機械学習と AI の専門知識と能力だった。AI と生物学が完全に一体化したのだ。
2 年後の「CASP14」にディープマインドのチームは再び参加した。サイエンティフィック・アメリカン誌は「生物学最大の問題のひとつがついに解決」と報じた。謎だったタンパク質の世界が驚異的なスピードで明らかにされた。AlphaFold2 があまりに優れていたため、CASP は役目を終えることになった。半世紀にわたってタンパク質の折り畳み問題は科学界最大の難問のひとつだったが、突然、難問リストから外された。
2022 年、AlphaFold2 は一般公開された。その結果、世界最先端の機械学習ツールが爆発的に流通し、生物学の基礎研究から応用研究まで、幅広く利用されることになった。ある研究者は、これは「地殻変動」だと表現した。公開からわずか 1 年半で 100 万人以上の研究者と、ほぼすべての主要な生物学研究機関が利用し、抗生物質への耐性、希少疾患の治療、生命の起源など、さまざまな問題に取り組んだ。
過去の実験で判明していたタンパク質の立体構造は欧州バイオインフォマティクス研究所（EBI）のデータベースに登録されていたが、その数は存在する既知のタンパク質のたった約 0.1％、19 万種に過ぎなかった。だがディープマインドは一気に約 2 億種のタンパク質の立体構造をアップロードしたのだ。これは既知のタンパク質をほぼ網羅したに等しい。かつての研究者たちはひとつのタンパク質の形状と機能を解明するのに数週間あるいは数カ月かけていたが、今や同じことが一瞬にして可能になったのだ。これがまさに指数関数的変化であり、来たるべき波がもたらし得ることだ。
だが、これはふたつのテクノロジーの融合の始まりに過ぎない。バイオ革命は AI とともに進化している。実際、本書で論じる多くの現象は AI があるから実現可能だ。ふたつの波がぶつかりあうことを考えてほしい。ひとつの波ではなく、巨大な波が生まれるのだ。
AI と合成生物学は、知性と生命という、極めて基本的で相互に関係している概念、人間の中核を成すふたつの特性を再構築し、操作するものだ。だから大局的見地から考えれば、AI と合成生物学は相互に置き換え可能な概念であり、同じプロジェクトだとわかる。
生物学は本当に複雑であり、タンパク質の立体構造のように、従来の技術ではほぼ解析不可能な莫大なデータが生み出される。その結果、新世代のツールが不可欠になった。開発チームは自然言語の指示だけで、新しい DNA 配列を生成するプロダクトを開発しようとしている。
ここでも Transformer（Google の研究者が開発した深層学習モデル）が人間には理解不能な長く複雑な DNA 配列に関係性と重要性を見出しながら、生物学や化学の言語を学習している。生化学データを追加学習した大規模言語モデルは、新しい分子やタンパク質、DNA や RNA の配列の有望な候補を生成する。その構造や機能、反応特性は実験室で検証する前からシミュレーションで予測できる。応用範囲と開発速度は増すばかりだ。
一部の科学者は、人間の頭脳をコンピュータに接続する方法を調査し始めている。2019 年には、まったく体を動かせない ALS（筋萎縮性側索硬化症）患者が脳に電極を埋め込む手術を受けた。電極は脳波を拾い、それが機械学習によって文字に置き換えられ、コンピュータ画面に「すばらしいわが子を愛している」と言葉を綴ることができた。
ニューラリンクのような企業は、脳と機械を直接つなぐブレイン・マシン・インターフェースに取り組んでいる。同社は 2021 年に、ブタの脳に人間の毛髪よりも細い 3000 本のフィラメント状の電極を埋め込み、ニューロンの活動をモニターした。すでに同社製の「N1」脳インプラントは、ヒト臨床試験も開始している。
シンクロン（Synchron）もオーストラリアでヒト臨床試験を開始した。コーティカル・ラブズ（Cortical Labs）というスタートアップ企業の科学者たちは、容器内で脳の一種を培養し（同社はこれをディッシュブレインと呼ぶ）、アタリゲーム「ポン」のプレイ方法を教えた。カーボンナノチューブ製の「神経レース」が、人間をデジタル世界に直接つなぐ日はそう遠くないかもしれない。
人間の知性が、瞬時にインターネットやクラウドと同じスケールの演算能力と情報量にアクセスできれば、何が起こるか？　ほとんど想像がつかないが、研究者たちはすでにその実現の初期段階にいる。来たるべき波の中心をなす汎用技術である AI と合成生物学は、すでに深く関わり合っており、相互に増強し合う、らせん型のフィードバックループを形成している。
新型コロナウイルスとパンデミックによって生命工学への一般的な認知度が高まった。だが、合成生物学の影響力の大きさは、可能性や危険性を含めて一般にはまったく知られていない。
バイオマシンとバイオコンピュータの時代へようこそ。この時代には DNA 鎖が計算を行い、人工細胞が作動する。機械は生命体になる。合成生命の時代へようこそ。
［日経 BOOK プラス 2024 年 10 月 9 日付の記事を転載］
