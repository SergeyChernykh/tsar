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

void initialize2() {
#pragma dvm actual(n2)
  for (int j = 0; j < 10; j++) {
#pragma dvm actual(n2)
#pragma dvm region in(n2)out(a)
    {
#pragma dvm parallel(1)
      for (int k = 0; k < n2; k++) {
        a[k] = k;
      }
    }
#pragma dvm get_actual(a)
  }
#pragma dvm get_actual(a)
}

void init() {
#pragma dvm actual(n, n2)
  initialize();
  initialize2();
#pragma dvm get_actual(a)
}

int main(void) {
#pragma dvm actual(n, n2)
  init();
#pragma dvm get_actual(a)
}