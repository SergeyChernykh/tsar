static int a[10];
int n = 10;
int n2 = 10;
void initialize() {
    for (int i = 0; i < n; i++) {
      a[i] = 0;
    }
}

int main(void) {
  for(int h = 0; h < 100; h++) {
    initialize();
  }
}