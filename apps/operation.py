import sys
import douban
import yelp
import movielens

execute = False
for item in sys.argv:
    # 檢查python檔是否正在執行訓練或預測，若是則強制離開
    if item == "list":
        print("douban")
        print("yelp")
        print("movielens")
        execute = True
    elif item == "douban":
        douban.main()
        execute = True
    elif item == "yelp":
        yelp.main()
        execute = True
    elif item =="movielens":
        movielens.main()
        execute = True

if not execute:
    print("start all process")
    douban.main()
    yelp.main()
    movielens.main()
