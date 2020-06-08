static int a[10];
int n = 10;

int main() {
  int s = 0;

#pragma dvm actual(n)
#pragma dvm region in(n)out(a)
  {
#pragma dvm parallel(1)
    for (int i = 0; i < n; ++i) {
      a[i] = i;
    }
  }
#pragma dvm get_actual(a)

  a[0] = 100;
#pragma dvm actual(a, n, s)
#pragma dvm region in(a, n, s)out(s)
  {
#pragma dvm parallel(1) reduction(sum(s))
    for (int j = 0; j < n; ++j) {
      s += a[j];
    }
  }
#pragma dvm get_actual(s)

  return s;
}