Week 6: reveal.js 公開用ファイル
=================================

【ローカル表示】
  Node.js がインストールされていれば start_week6.bat をダブルクリック。
  あるいはこのフォルダで: node server_week6.js
  http://localhost:8000/week6.html にアクセスします。
  すでにポート8000を使用している場合は、別のサーバーを止めるか
  PowerShellで $env:PORT=8001; node server_week6.js と実行してください。
  ※ index.html をダブルクリックする file:// 方式では、外部Markdownを読み込めません。

【GitHub Pages】
  既存の「Rindoku」リポジトリ (main ブランチのルート) に、
    week6.html
    slides/6_multiclass_classification.md
    images/multiclass_classification/ の画像すべて
    css/week6.css
  を同じ相対位置に追加します（zipファイルそのものを置くのではありません）。
  すでにweek2がある場合、slides/images/cssのフォルダにファイルを追加してください。
  week2.htmlとcss/custom.cssは変更しません。
  公開URL: https://USERNAME.github.io/Rindoku/week6.html
  （USERNAMEは自分のGitHubユーザー名）

【仕様】
  横移動 --- / 縦移動 -- はMarkdown内の単独行で区切ります。
  KaTeXで $...$ / $$...$$ の数式を表示します。
  8枚のローカルPNGは元のimagesフォルダから読みます。
  MNISTの例示画像1枚はWikimedia Commonsから読みます。
  reveal.js / KaTeXはCDN配信なので、閲覧時にインターネット接続が必要です。
  画像をスライド内に収めるCSSをweek6専用ファイルで指定しています。
  全スライドの縦中央配置 center: true は維持しています。
