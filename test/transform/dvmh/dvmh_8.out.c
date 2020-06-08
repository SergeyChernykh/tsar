static int a[10];
int n = 10;
int n2 = 10;
void initialize() {
#pragma dvm actual(n)
#pragma dvm region in(n)out(a)
  {
#pragma dvm parallel(1)
    for (int i = 0; i < n; i++) {
      a[i] = 0;
    }
  }
#pragma dvm get_actual(a)
}

int main(void) {
#pragma dvm actual(n)
  for (int h = 0; h < 100; h++) {
#pragma dvm actual(n)
    initialize();
#pragma dvm get_actual(a)
  }
#pragma dvm get_actual(a)
}