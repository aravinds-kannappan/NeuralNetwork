from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json
from playwright.sync_api import sync_playwright
R=Path('/Users/aravindkannappan/.local/share/openresearch/files/neuralnetworks/abay-review/geometry-transfer')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
p=Links();p.feed((R/'report.html').read_text());assert len(p.ids)==len(set(p.ids))
for href in p.links:
    u=urlsplit(href)
    if u.scheme:continue
    if u.path:assert (R/unquote(u.path)).exists(),href
    elif u.fragment:assert u.fragment in p.ids,href
errors=[];checks=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((R/'report.html').as_uri(),wait_until='load')
    assert page.locator('h1').inner_text()=='One frozen layer,\nfive robot policies'
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='/tmp/geometry-walkthrough-desktop.png')
    assert page.locator('#success-bars .bar-row').count()==4
    assert page.locator('#success-bars output').all_text_contents()==['30/60 · 50.0%','30/60 · 50.0%','26/60 · 43.3%','11/60 · 18.3%']
    page.click('[data-group="scores"][data-value="shift"]')
    assert page.locator('#success-bars output').all_text_contents()==['9/60 · 15.0%','9/60 · 15.0%','8/60 · 13.3%','9/60 · 15.0%']
    page.check('#show-intervals');assert page.locator('#interval-note').is_visible()
    page.click('[data-group="scores"][data-value="id"]')
    page.locator('#comparison').scroll_into_view_if_needed();page.screenshot(path='/tmp/geometry-walkthrough-comparison.png')
    page.uncheck('#show-intervals')
    assert page.locator('.episode').count()==60
    assert [page.locator('#count-'+k).inner_text() for k in ['both','onlyp','onlyr','neither']]==['18','12','12','18']
    page.locator('.episode.onlyr').first.click();assert 'Pretrained: failure' in page.locator('#episode-readout').inner_text()
    page.click('[data-group="episodes"][data-value="shift"]')
    assert [page.locator('#count-'+k).inner_text() for k in ['both','onlyp','onlyr','neither']]==['1','8','8','43']
    assert '17/60' in page.locator('#union-note').inner_text()
    page.click('[data-group="episodes"][data-value="id"]')
    page.locator('#ch2 .figure').scroll_into_view_if_needed();page.screenshot(path='/tmp/geometry-walkthrough-paired.png')
    page.click('[data-group="audit"][data-value="1000"]');assert page.locator('#audit-centered').inner_text()=='0.98050'
    page.click('[data-group="audit"][data-value="100"]');assert page.locator('#audit-centered').inner_text()=='0.99797'
    for key in ['state','encoder','frozen','decoder','robot']:
        page.click('[data-flow="'+key+'"]');assert page.locator('#flow-readout').inner_text()
    page.click('[data-flow="state"]')
    page.click('#technical-toggle');assert page.locator('details.technical[open]').count()==page.locator('details.technical').count()
    page.click('#technical-toggle');assert page.locator('details.technical[open]').count()==0
    page.click('#replay-toggle');assert page.locator('#recorded-replay').get_attribute('src').startswith('data:image/gif;')
    page.click('#replay-toggle');assert page.locator('#recorded-replay').get_attribute('src').startswith('data:image/png;')
    page.click('#theme-toggle');assert page.locator('html').get_attribute('data-theme')=='dark'
    page.locator('#comparison').scroll_into_view_if_needed();page.screenshot(path='/tmp/geometry-walkthrough-dark.png')
    page.click('#theme-toggle')
    page.set_viewport_size({'width':390,'height':844})
    for section in ['main','words','ch0','comparison','ch2','ch3','ch4','ch5','ch7']:
        page.locator('#'+section).scroll_into_view_if_needed()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),section
    page.locator('#comparison').scroll_into_view_if_needed();page.screenshot(path='/tmp/geometry-walkthrough-mobile.png')
    page.locator('#words').scroll_into_view_if_needed();page.screenshot(path='/tmp/geometry-walkthrough-terms-mobile.png')
    assert not errors,errors
    browser.close()
checks={'local_links_checked':len(p.links),'unique_ids':len(p.ids),'desktop_and_mobile_overflow':'pass','success_filters_and_counts':'pass','episode_grid_and_pair_counts':'pass','checkpoint_switch':'pass','architecture_controls':'pass','technical_note_controls':'pass','theme_toggle':'pass','saved_replay_controls':'pass','browser_errors':errors,'new_experiments':0}
(R/'presentation-checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
