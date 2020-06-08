static int a[10];
int n = 10;

int main() {
#pragma dvm actual(n)
#pragma dvm region in(n)out(a)
  {
#pragma dvm parallel(1)
    for (int i = 0; i < n; ++i) {
      a[i] = i;
    }
  }
  int k = 0;
#pragma dvm region in(n)out(a)
  {
#pragma dvm parallel(1)
    for (int j = 0; j < n; ++j) {
      a[j] = 1;
    }
  }
#pragma dvm get_actual(a)

  return 0;
}