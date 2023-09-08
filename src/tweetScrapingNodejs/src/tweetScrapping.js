import puppeteer from 'puppeteer';
import fs from 'fs';

const scrapping = async (query, minTweets) => {
    // Lanzamos un navegador
    const browser = await puppeteer.launch({ headless: false });

    // Creamos una pestaña
    const page = await browser.newPage();

    // Navegar hacia el sitio
    await page.goto(`https://twitter.com/search?q=${query}&f=live`);

    // Espero por el input del mail
    const inputMail = await page.waitForSelector('.r-30o5oe.r-1niwhzg.r-17gur6a.r-1yadl64.r-deolkf.r-homxoj.r-poiln3.r-7cikom.r-1ny4l3l.r-t60dpp.r-1dz5y72.r-fdjqy7.r-13qz1uu');
    // Escribo en el input
    await inputMail.type("Conan1339017", { delay: 100 });

    // Presiono enter
    await page.keyboard.press('Enter');

    // Espero por el input de password
    const inputPassword = await page.waitForSelector('.r-30o5oe.r-1niwhzg.r-17gur6a.r-1yadl64.r-deolkf.r-homxoj.r-poiln3.r-7cikom.r-1ny4l3l.r-t60dpp.r-1dz5y72.r-fdjqy7.r-13qz1uu', { visible: true });
    // Escribo en el input
    await inputPassword.type("mileiScraping", { delay: 100 });

    // Presiono enter
    await page.keyboard.press('Enter');

    // Espero por los tweets
    await page.waitForSelector('[data-testid="tweetText"]')

    let dataSet = []

    for (let i = 0; i < minTweets / 10; i++) {
        let tweets = await page.$$eval('[data-testid="tweetText"]', spans => {
            return spans.map(span => span.textContent)
        })
        dataSet = [...dataSet, ...tweets]

        //await page.evaluate('window.scrollBy(0,2000)');
        await page.mouse.wheel({ deltaY: 3000 });
        await new Promise(r => setTimeout(r, 1500));
    }

    //console.log(dataSet)
    console.log(dataSet.length)

    // Elimino repetidos
    const dataSetConjunto = new Set(dataSet)
    console.log("Tweets obtenidos: " + dataSetConjunto.size)

    // Cerrar el navegador
    await browser.close();

    return [...dataSetConjunto]
}

const armarQuery = (query) => {
    return query.replace(" ", "%20")
}

const armarCSV = (data) => {

    let tweetsCSV = "Tweets;\n";

    data.forEach(tweet => {

        let tweetSinSaltoLinea = tweet.replace(/\n/g, " ")
        tweetsCSV = tweetsCSV.concat(tweetSinSaltoLinea + ";\n")

    });

    fs.writeFileSync("tweets.csv", tweetsCSV);
}

const main = (async () => {

    const query = "Milei"
    const minTweets = 100

    const data = await scrapping(armarQuery(query), minTweets);

    armarCSV(data)
})();
