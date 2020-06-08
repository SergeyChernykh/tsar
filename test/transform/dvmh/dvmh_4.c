static int a[10];
int n = 10;

int main() {
  for (int i = 0; i < n; ++i) {
    a[i] = i;
  }
  int k = 0;
  for (int j = 0; j < n; ++j) {
    a[j]= 1;
  }
  return 0;
}