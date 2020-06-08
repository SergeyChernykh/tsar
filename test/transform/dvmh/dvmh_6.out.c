static int a[10];
int n = 10;

int main() {
  int s = 0;
#pragma dvm actual(a, n, s)
  for (int k = 0; k < 10; k++) {
#pragma dvm actual(a, n, s)
#pragma dvm region in(a, n, s)out(a, s)
    {
#pragma dvm parallel(1)
      for (int i = 0; i < n; ++i) {
        a[i] = i;
      }

#pragma dvm parallel(1) reduction(sum(s))
      for (int j = 0; j < n; ++j) {
        s += a[j];
      }
    }
#pragma dvm get_actual(a, s)
  }
  for (int k = 0; k < 10; k++) {
#pragma dvm actual(a, n, s)
#pragma dvm region in(a, n, s)out(a, s)
    {
#pragma dvm parallel(1)
      for (int i = 0; i < n; ++i) {
        a[i] = i;
      }

#pragma dvm parallel(1) reduction(sum(s))
      for (int j = 0; j < n; ++j) {
        s += a[j];
      }
    }
#pragma dvm get_actual(a, s)
  }
#pragma dvm get_actual(a, s)

  return s;
}