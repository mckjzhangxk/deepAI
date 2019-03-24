package recfun

object Main {
  def main(args: Array[String]) {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(pascal(col, row) + " ")
      println()
    }
//    println(balance("())(".toList))
  }

  /**
   * Exercise 1
   */
    def pascal(c: Int, r: Int): Int = {
      if (c==0 ||c==r) 1 else pascal(c-1,r-1)+pascal(c,r-1)
    }
  
  /**
   * Exercise 2
   */
    def balance(chars: List[Char]): Boolean = {
      var cnt = 0

      for(x<-chars){
        if(x=='(') cnt+=1
        if(x==')') cnt-=1
        if(cnt<0)
          return false
      }
      if(cnt==0) true else false
    }
  
  /**
   * Exercise 3
   */
    def countChange(money: Int, coins: List[Int]): Int = ???
  }
