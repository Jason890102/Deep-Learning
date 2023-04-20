一、 將檔案放從本機放至雲端
  1. git add -A &emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;&nbsp;# 將更動檔案新增至暫存區  
  2. git commit -m'name' &nbsp;# 命名版本名稱  
  3. git push   &emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 將更動檔案放至雲端  
--- 
二、 將檔案放從雲端至本機
  1. git clean -d -fx  &nbsp;&nbsp;&nbsp;&nbsp;# d:刪除未被添加到git的路徑上的文件、f:刪除忽略文件已經對git來說不識別的文件、f:強制執行
  2. git pull   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 將更動檔案放至本機
---  
三、 其他指令  
  1. 換行: 按兩個空格  
  2. 
