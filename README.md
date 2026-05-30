# Reper Kalibrlash

Nivelirlash tarmog'idagi reperlarni tenglashtirish va kalibrlash uchun dastur.

## Ishga tushirish

```bash
pip install numpy pyqt6
python kalibrlash.py
```

## Foydalanish

1. Reperlar ma'lumotlarini jadvalga kiriting
2. "Holat" ustunida reperni `Turg'un` (dizayn balandligi o'zgarmas) yoki `Sozlanadi` deb belgilang
3. Kamida 1 ta `Turg'un` reper bo'lishi kerak
4. **Hisoblash** tugmasini bosing
5. Natijalarni **Saqlash** bilan faylga yozib oling

> Holatni o'zgartirish uchun katakka ikki marta bosing.

## Hisoblash usuli

Dastur eng kichik kvadratlar usulida (EKKU) tarmoq tenglashtirish bajaradi:

- **Kuzatishlar**: qo'shni reperlar orasidagi balandlik farqlari
- **Noma'lumlar**: sozlanadigan reperlarning eng ehtimoliy balandliklari
- **Og'irliklar**: masofaga bog'liq (σ ~ 2mm/√km)
- **Turg'un reperlar**: tenglashtirishda o'zgarmas qatnashadi

Natijada har bir reperning tuzatilgan balandligi, standart og'ishi va
qo'pol xatolar statistikasi chiqariladi.
