//#############################################################
// const
// https://www.youtube.com/watch?v=7arYbAhu0aw&list=PLE28375D4AC946CC3&index=1
//   - A compile time constraint that an object can not be modified
//
int main(int argc, char const *argv[])
{
    const int i=0;
    // i=0;  
    // If const is on the left  of *, data is const
    // If const is on the right of *, pointer is const
    const int* p1=&i;
    // *p1=2;
    p1=p1+1;

    int b=2;
    int* const p2=&b;
    *p2=11;
    // p2=p2+1;

    const int* const p3=&i;

    int const * p4;//const int *p4
    // *p4=1;
    return 0;
}
