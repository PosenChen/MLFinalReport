使用方法：
1. 至36行修改參數(字串)，以決定模型的架構。 ex: CRPF = "CRCRPF" 或 CRPF = "CRPCRPFF" 或 CRPF = "CRPCRCRPCRCRCRPFFF" ... 。
2. 依上一點所修改的參數中字母C,R,P的總數量，來決定要在第153行的self.block_1 = nn.Sequential(  ) 的括弧中依次輸入幾個物件。 ex: CRPF = "CRCRPF" 的CRP共5個 故括弧中輸入 sb[0],sb[1],sb[2],sb[3],sb[4] 。
3. 依上一點所修改的參數中字母F的總數量+1，來決定要在第154行的self.classifier  = nn.Sequential(  ) 的括弧中依次輸入幾個物件。 ex: CRPF = "CRCRPF" 的F共1個 故括弧中輸入 sc[0],sc[1] 。
4. 執行全部的程式碼，依畫面指示手動輸入out_channels= ,kernel_size=, stride=, padding= 等參數。(要先將想要跑的架構先想好，輸入的參數有點多，建議將要設定的參數先寫在紙上...)
5. 程式執行完畢後會將結果更新在工作目錄下的 "CNN結果輸出.csv"檔案。
